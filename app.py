import streamlit as st
import os
import shutil
import zipfile
import requests
import uuid
import stat

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_cohere import ChatCohere
from langchain_classic.chains import RetrievalQA







st.set_page_config(
    page_title="Codebase Assistant",
    layout="wide",
)
st.markdown("""
<style>
    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #f5f5f5;
    }
    .app-subtitle {
        color: #9aa0ab;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .stApp {
    background: linear-gradient(180deg, #0e1117 0%, #131720 100%);
    }

    section[data-testid="stSidebar"] {
        background-color: #10141c;
        border-right: 1px solid #232a36;
    }
    .source-chip {
    display: inline-block;
    background-color: #1f2530;
    color: #7fd1a5;
    border: 1px solid #2c3340;
    border-radius: 6px;
    padding: 2px 10px;
    margin: 3px 4px 0 0;
    font-size: 0.8rem;
    font-family: monospace;
    }
    .status-box {
        background-color: #161b24;
        border: 1px solid #262d3a;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 10px;
        color: #c8ccd4;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='app-title'>Github Codebase Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Ask questions about any public GitHub repo and get grounded answers with citations.</div>", unsafe_allow_html=True)

def build_pipeline(repo_url, cohere_api_key):

    def remove_readonly(func, path, excinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    work_dir = "./repo_download"
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir, onerror=remove_readonly)
    os.makedirs(work_dir)

    # download
    repo_url_clean = repo_url.rstrip("/").removesuffix(".git")
    owner_repo = "/".join(repo_url_clean.split("/")[-2:])   # <-- must come BEFORE the loop below

    zip_path = os.path.join(work_dir, "repo.zip")
    downloaded = False

    for branch in ("main", "master"):
        url = f"https://github.com/{owner_repo}/archive/refs/heads/{branch}.zip"
        resp = requests.get(url)
        if resp.status_code == 200:
            with open(zip_path, "wb") as f:
                f.write(resp.content)
            downloaded = True
            break

    if not downloaded:
        st.error(f"Could not download repo from {owner_repo}. Check the URL is correct and the repo is public.")
        st.stop()



    # extract
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(work_dir)
    extracted = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
    repo_path = os.path.join(work_dir, extracted[0])

    # read files (your guardrails version)
    ALLOWED_EXTENSIONS = {".py", ".md", ".js", ".json"}
    IGNORED_DIRS = {"__pycache__", ".git", "node_modules", "venv"}
    MAX_FILE_SIZE_BYTES = 200_000

    documents = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext not in ALLOWED_EXTENSIONS:
                continue
            filepath = os.path.join(root, filename)
            if os.path.getsize(filepath) > MAX_FILE_SIZE_BYTES:
                continue
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                continue
            if not text.strip():
                continue
            rel_path = os.path.relpath(filepath, repo_path)
            documents.append(Document(page_content=text, metadata={"source": rel_path}))

    # chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # embed + store


# instead of a fixed "./chroma_db" folder, make a new one each time:
    chroma_dir = f"./chroma_db_{uuid.uuid4().hex[:8]}"    
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_dir)

    # build chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatCohere(cohere_api_key=cohere_api_key, model="command-a-03-2025")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    return qa_chain, len(documents), len(chunks)


# actual UI 


with st.sidebar:
    cohere_api_key = st.text_input("Cohere API Key", type="password")
    repo_url = st.text_input("GitHub Repo URL", placeholder="https://github.com/owner/repo")
    build_clicked = st.button("Load Repository")

    if "stats" in st.session_state:
        st.markdown(
            f"<div class='status-box'>✅ {st.session_state['stats']['files']} files loaded<br>"
            f"✅ {st.session_state['stats']['chunks']} chunks indexed</div>",
            unsafe_allow_html=True,
        )

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if build_clicked:
    if not cohere_api_key or not repo_url:
        st.error("Enter both a Cohere API key and a repo URL.")
    else:
        with st.spinner("Building pipeline... this takes a minute."):
            qa_chain, n_files, n_chunks = build_pipeline(repo_url, cohere_api_key)
            st.session_state.qa_chain = qa_chain
            st.session_state.stats = {"files": n_files, "chunks": n_chunks}  # NEW
        st.success(f"Ready! Indexed {n_files} files, {n_chunks} chunks.")


#For prev chats
 
if "messages" not in st.session_state:
    st.session_state.messages = []





#getting ans
if st.session_state.qa_chain is None:
    st.markdown(
        "<div class='status-box'>Load a GitHub repo from the sidebar to get started...</div>",
        unsafe_allow_html=True,
    )
else:
    st.success("Pipeline ready — ask a question below.")

    user_question = st.chat_input("Ask something about this codebase...")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                st.markdown(
                "".join(f"<span class='source-chip'>{s}</span>" for s in msg.get("sources")),
                unsafe_allow_html=True,
                )

    if user_question:
        st.session_state.messages.append({'role':'user','content':user_question})
        with st.chat_message("user"):
            st.write(user_question)
    

        

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
           
                response = st.session_state.qa_chain.invoke({"query": user_question})
              
                answer = response["result"]
                sources = sorted(set(doc.metadata["source"] for doc in response["source_documents"]))
                st.write(answer)
                st.markdown(
                "".join(f"<span class='source-chip'>{s}</span>" for s in sources),
                unsafe_allow_html=True,
                )


            st.session_state.messages.append({'role':'assistant','content':answer,'sources':sources})
