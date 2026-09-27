import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda,RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

from langsmith import traceable  # <-- key import


"""
@tracable is used to track every method in the traces
"""

PDF_PATH = "8_Langsmith/islr.pdf"
load_dotenv()

# ---------- traced setup steps ----------

@traceable(name="load_pdf")
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return docs


@traceable(name="split_docs")
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150
    )

    splits = splitter.split_documents(docs)
    return splits


@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        splits,
        embeddings
    )

    return vectorstore


# ---------- You can also trace a “setup” umbrella span ----------

@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_docs(docs)
    vs = build_vectorstore(splits)
    return vs




#================    PIPELINE    =====================
os.environ["LANGCHAIN_PROJECT"] = "Rag with tracable"

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)


prompt = ChatPromptTemplate([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])


# Build the index under traced setup
vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 4}
    )
    
#join docs
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


parallel = RunnableParallel({
    "context": RunnableSequence(retriever,RunnableLambda(format_docs)),
    "question": RunnablePassthrough()
})


chain = parallel | prompt | model | StrOutputParser()

# ---------- run a query (also traced) ----------
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ").strip()

# Give the visible run name + tags/metadata so it’s easy to find:
config = {
    "run_name": "pdf_rag_query"
}

ans = chain.invoke(q, config=config)
print("\nA:", ans)