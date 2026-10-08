from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded:", embeddings.model_name)

vector_store = FAISS.load_local(
    "vector_store", embeddings, allow_dangerous_deserialization=True)
print("Vector store loaded from local storage.")

query = "Tell me about pets."
results = vector_store.similarity_search(query, k=2)
print("Query:", query)
print("Top 2 similar texts:")
for r in results:
    print("-", r.page_content)
