import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings

def compute_embeddings(chunks):
    """
    Computes embeddings for each text chunk using OpenAIEmbeddings.
    Returns a NumPy array of embeddings and the embedding model instance.
    """
    embedding_model = OpenAIEmbeddings()
    embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]
    return np.array(embeddings), embedding_model

def cosine_similarity(query_emb, embeddings):
    """Computes cosine similarity between a query embedding and each chunk embedding."""
    query_norm = query_emb / np.linalg.norm(query_emb)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.dot(embeddings_norm, query_norm)

def retrieve_chunks(query, chunks, embeddings, embedding_model, top_k=3):
    """
    Retrieves the top_k chunks most similar to the query using cosine similarity.
    """
    query_emb = np.array(embedding_model.embed_query(query))
    sims = cosine_similarity(query_emb, embeddings)
    top_indices = sims.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]