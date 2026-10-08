import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded:", embeddings.model_name)


def similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


vector1 = embeddings.embed_query("I like dogs.")
vector2 = embeddings.embed_query("I like cats.")
vector3 = embeddings.embed_query("I enjoy programming.")
sim12 = similarity(vector1, vector2)
sim13 = similarity(vector1, vector3)
sim23 = similarity(vector2, vector3)
print("Similarity between 'I like dogs.' and 'I like cats.':", sim12)
print("Similarity between 'I like dogs.' and 'I enjoy programming.':", sim13)
print("Similarity between 'I like cats.' and 'I enjoy programming.':", sim23)
