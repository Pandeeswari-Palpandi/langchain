from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

texts = [
    "Python is a programming language.",
    "Streamlit is a framework for building web apps.",
    "LangChain is a framework for building applications with LLMs.",
    "Dogs are loyal animals.",
    "Cats are independent animals."]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded:", embeddings.model_name)

vector_store = FAISS.from_texts(texts, embeddings)
print("Vectors generated for each text.")

vector_store.save_local("vector_store")

query = "Tell me about programming languages."
results = vector_store.similarity_search(query, k=2)
print("Query:", query)
print("Top 2 similar texts:")
for r in results:
    print("-", r.page_content)
