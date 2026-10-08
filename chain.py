from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

prompt = ChatPromptTemplate.from_template(
    "Explain the following text in simple terms: {text}")

model = ChatOpenAI(model_name="gpt-4o-mini",
                   temperature=0, openai_api_key=api_key)

ouput_parser = StrOutputParser()

chain = prompt | model | ouput_parser
print("chain created:", chain)

answer = chain.invoke({"text": "langChain"})
print("Answer:", answer)
