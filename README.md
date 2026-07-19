# GitHub Codebase Assistant

A RAG-powered chat tool that lets developers ask natural-language questions about any public GitHub repository and get answers grounded in the actual source code, with file-level citations.

## Problem

Onboarding onto an unfamiliar codebase is slow. New developers — interns, new hires, or contributors to open-source projects — typically spend days manually reading through files, folder structures, and documentation just to answer simple questions like "how does authentication work here?" or "where do I add a new route?" This tool shortens that ramp-up time by letting people ask these questions directly and get grounded, source-linked answers instead of searching manually.

## How It Works

1. **Repo Ingestion** — The user provides a public GitHub repo URL. The app downloads it as a zip archive (no authentication needed), extracts it, and reads all code and documentation files, skipping irrelevant directories (e.g. `node_modules`, `.git`) and oversized or non-text files.

2. **Chunking** — Each file's content is split into smaller, overlapping text chunks (~1000 characters) using LangChain's `RecursiveCharacterTextSplitter`, so retrieval can return precise, relevant sections rather than entire files.

3. **Embedding & Indexing** — Each chunk is converted into a vector embedding (via a local HuggingFace sentence-transformer model) and stored in a Chroma vector database, along with metadata tracking which file it came from.

4. **Retrieval-Augmented Generation** — When a user asks a question, the top-matching chunks are retrieved from the vector store and passed to Cohere's `command-a-03-2025` chat model, which generates an answer grounded in that retrieved code.

5. **Citations** — Every answer displays the specific file(s) it was generated from, so users can verify the answer or dig deeper into the actual source.