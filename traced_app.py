from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT_NAME")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

print("LangSmith is connected!")

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
questions = [
    "Explain the concept of LangChain in simple terms.",
    "What are the key features of LangChain?",
    "How can LangChain be used in real-world applications?",
]
chain = prompt | model | StrOutputParser()

for question in questions:
    answer = chain.invoke({"question": question})
    print(f"Question: {question}\nAnswer: {answer}\n")
