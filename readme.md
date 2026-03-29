# RAG Pipeline — LangChain Edition

A Retrieval-Augmented Generation (RAG) pipeline rebuilt using LangChain. This version replaces the manual embedding and ChromaDB setup from the previous iteration with LangChain's document loaders, text splitters, and vector store abstractions — resulting in cleaner code and persistent vector storage across runs. This iteration further introduces LCEL (LangChain Expression Language) to replace manual prompt construction with a declarative, composable chain.

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
User Question
      |
      v
LCEL Chain:
  retriever | format_docs  -->  {context}
  RunnablePassthrough()    -->  {question}
      |
      v
  ChatPromptTemplate
      |
      v
  ChatGroq (LLaMA 3.3 70B)
      |
      v
  StrOutputParser()
      |
      v
  Answer printed to terminal
```

---

## What Changed from the Previous Version

| | Previous (manual) | This version (LangChain + LCEL) |
|---|---|---|
| Document loading | `open()` + `read()` | `TextLoader` |
| Chunking | `split("\n\n")` | `RecursiveCharacterTextSplitter` (size + overlap) |
| Embeddings | `SentenceTransformer` directly | `HuggingFaceEmbeddings` via LangChain |
| Vector store | `chromadb.Client()` (in-memory) | `Chroma` with `persist_directory` (persistent) |
| Retriever | `similarity_search()` called directly | `vector_store.as_retriever()` |
| Prompt construction | f-string manually | `ChatPromptTemplate.from_template()` |
| LLM call | `groq.Client()` manually | `ChatGroq` LangChain integration |
| LLM output | `.content` attribute | `StrOutputParser()` |
| Pipeline execution | Manual step-by-step calls | LCEL chain (`\|` operator) |

---

## Tech Stack

- **LangChain** - Orchestration framework for the RAG pipeline
- **LangChain Core** - `ChatPromptTemplate`, `StrOutputParser`, `RunnablePassthrough` for LCEL
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
- `as_retriever()` for wrapping the vector store as a LangChain-compatible retriever
- LCEL (LangChain Expression Language) and the `|` pipe operator for composing chains
- `ChatPromptTemplate` for structured, reusable prompt templates
- `StrOutputParser` for extracting plain string output from the LLM
- `RunnablePassthrough` for passing the question through the chain unchanged
- Using `ChatGroq` as a LangChain-compatible LLM
- Grounded prompting with an explicit fallback ("I don't know based on the provided document")

---

## Limitations and Improvements

- **Single document only**: The pipeline loads one hardcoded file (`example.txt`). Extending it to accept dynamic file paths or multiple documents would make it more flexible.
- **No deduplication on re-run**: Running the script multiple times on the same file will add duplicate chunks to the vector store. Adding a check before inserting would prevent this.
- **CLI only**: The pipeline runs in the terminal. Wrapping it in a FastAPI endpoint would make it ready for integration with a frontend or other services.
- **Fixed chunk size**: `chunk_size=500` and `chunk_overlap=50` are hardcoded. Making these configurable via environment variables or CLI arguments would allow better tuning per document type.