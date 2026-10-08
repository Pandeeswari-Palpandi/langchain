from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", openai_api_key=api_key)
print("OpenAI Embedding model loaded:", openai_embeddings.model)

text = "Python is a popular programming language known for its simplicity and readability."
vector = openai_embeddings.embed_query(text)
print("Vector representation of the text:", vector)
print("Length of the vector:", len(vector))
print("First 10 elements of the vector:", vector[:10])
