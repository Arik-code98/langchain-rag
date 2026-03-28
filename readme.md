# RAG Pipeline — LangChain Edition

A Retrieval-Augmented Generation (RAG) pipeline rebuilt using LangChain. This version replaces the manual embedding and ChromaDB setup from the previous iteration with LangChain's document loaders, text splitters, and vector store abstractions — resulting in cleaner code and persistent vector storage across runs.

---

## How It Works

```
example.txt
      |
      v
TextLoader  -->  RecursiveCharacterTextSplitter
                 (chunk_size=500, overlap=50)
      |
      v
HuggingFaceEmbeddings (all-MiniLM-L6-v2)
      |
      v
Chroma Vector Store  (persisted to ./chroma_langchain_db)
      |
      v
User Question  -->  similarity_search (k=2)
                          |
                          v
                  Top-2 Relevant Chunks
                          |
                          v
              Prompt = Context + Question
                          |
                          v
          LLaMA 3.3 70B via ChatGroq (Groq API)
                          |
                          v
                   Answer printed to terminal
```

---

## What Changed from the Previous Version

| | Previous (manual) | This version (LangChain) |
|---|---|---|
| Document loading | `open()` + `read()` | `TextLoader` |
| Chunking | `split("\n\n")` | `RecursiveCharacterTextSplitter` (size + overlap) |
| Embeddings | `SentenceTransformer` directly | `HuggingFaceEmbeddings` via LangChain |
| Vector store | `chromadb.Client()` (in-memory) | `Chroma` with `persist_directory` (persistent) |
| LLM call | `groq.Client()` manually | `ChatGroq` LangChain integration |

---

## Tech Stack

- **LangChain** - Orchestration framework for the RAG pipeline
- **LangChain Community** - `TextLoader` for document ingestion
- **LangChain HuggingFace** - Embedding model wrapper (`all-MiniLM-L6-v2`)
- **LangChain Chroma** - Vector store integration with persistence
- **LangChain Groq** - `ChatGroq` LLM wrapper
- **ChromaDB** - Underlying vector database (persistent)
- **Groq API** - LLM inference backend
- **LLaMA 3.3 70B Versatile** - The underlying large language model
- **python-dotenv** - Secure API key management

---

## Project Structure

```
project/
├── main.py                  # RAG pipeline script
├── example.txt              # Source document to query against
├── chroma_langchain_db/     # Persisted ChromaDB vector store (auto-created)
├── .env                     # Environment variables (not committed)
└── requirements.txt         # Project dependencies
```

---

## Setup and Installation

**1. Clone the repository**

```bash
git clone https://github.com/Arik-code98/langchain-rag.git
cd langchain-rag
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment variables**

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key from [https://console.groq.com](https://console.groq.com).

**4. Add your document**

Place your source text in `example.txt`. The `RecursiveCharacterTextSplitter` handles chunking automatically — no special formatting required.

**5. Run the pipeline**

```bash
python main.py
```

You will be prompted to enter a question. The answer will be printed to the terminal.

---

## Key Concepts Explored

- Document loading and chunking with LangChain abstractions
- `RecursiveCharacterTextSplitter` with configurable `chunk_size` and `chunk_overlap`
- Persistent vector storage with ChromaDB's `persist_directory`
- Similarity search using LangChain's `Chroma.similarity_search()`
- Using `ChatGroq` as a LangChain-compatible LLM
- Grounded prompting with an explicit fallback ("I don't know based on the provided document")
- Replacing manual boilerplate with LangChain's higher-level API

---

## Limitations and Improvements

- **Single document only**: The pipeline loads one hardcoded file (`example.txt`). Extending it to accept dynamic file paths or multiple documents would make it more flexible.
- **No deduplication on re-run**: Running the script multiple times on the same file will add duplicate chunks to the vector store. Adding a check before inserting would prevent this.
- **CLI only**: The pipeline runs in the terminal. Wrapping it in a FastAPI endpoint would make it ready for integration with a frontend or other services.
- **Basic prompt**: The prompt instructs the model to stay within the context but does no reranking or relevance filtering. Adding a reranker would improve answer quality on longer documents.
- **Fixed chunk size**: `chunk_size=500` and `chunk_overlap=50` are hardcoded. Making these configurable via environment variables or CLI arguments would allow better tuning per document type.