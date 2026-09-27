from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

os.environ["LANGCHAIN_PROJECT"] = "sequential-project-demo"


load_dotenv()

os

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser


#to provide additional metadata in tracing
# Additional information for LangSmith tracing
config = {
    "run_name": "UnemploymentReportPipeline",
    "tags": [
        "qwen",
        "text-generation",
        "sequential-chain"
    ],
    "metadata": {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "task": "text_generation",
        "project": "sequential-project-demo",
        "chain_type": "sequential",
        "topic": "Unemployment in India",
        "environment": "development"
    }
}


result = chain.invoke({'topic': 'Unemployment in India'},config=config)

print(result)