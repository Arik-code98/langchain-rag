from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

loader=TextLoader("example.txt")
docs=loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

vector_store.add_documents(all_splits)

question = input("Ask a question: ")
results = vector_store.similarity_search(question, k=2)

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

context = "\n\n".join([doc.page_content for doc in results])
prompt = f"""Answer the question based ONLY on the context below. 
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}"""

response = llm.invoke(prompt)
print(response.content)