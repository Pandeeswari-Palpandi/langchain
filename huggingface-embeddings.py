from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded:", embeddings.model_name)

text = "Python is a popular programming language known for its simplicity and readability."
vector = embeddings.embed_query(text)
print("Vector representation of the text:", vector)
print("Length of the vector:", len(vector))
print("First 10 elements of the vector:", vector[:10])
