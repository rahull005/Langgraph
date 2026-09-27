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


"""
    here when we trace the rag application, observe that it wont trace all the application,
    like :
            chunks,
            splits,
            embeddings

    it only trace :
                    chain = parallel | prompt | llm | parser

                    -> it only tracs the chain

"""


load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Rag with trace"

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

PDF_PATH = "8_Langsmith/islr.pdf"

#1) load_pdf
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()


#2) chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)


#3) splitting
splits = splitter.split_documents(docs)

#4) embeddings
embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

vs = FAISS.from_documents(
    splits,
    embeddings
)


retriever = vs.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 4}
)


# 5) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])


# 6) format docs
def format_docs(docs): 
    return "\n\n".join(d.page_content for d in docs)


#7) prompt
parallel = RunnableParallel({
    "context": RunnableSequence(retriever,RunnableLambda(format_docs)),
    "question" : RunnablePassthrough()
})


#8) parser
parser = StrOutputParser()


#9) chaining
chain = RunnableSequence(parallel, prompt, model, parser)

#10) invoking
user_input = input("Enter the prompt here : ")
res = chain.invoke(user_input)

print(res)
