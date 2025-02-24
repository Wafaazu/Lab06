import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load precomputed document embeddings and documents
embeddings = np.load("embeddings.npy")
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_top_k(query_embedding, embeddings, k=10):
    """
    Retrieve the top k most similar documents using cosine similarity.
    """
    # Compute cosine similarities between the query and all document embeddings
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    # Get indices for the top k scores
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

# Placeholder function for generating a query embedding
def get_query_embedding(query):
    # For demonstration, returns a random embedding vector.
    # Replace with your actual embedding model when available.
    return np.random.rand(embeddings.shape[1])

# Streamlit UI
st.title("Information Retrieval using Document Embeddings")
query = st.text_input("Enter your query:")

if st.button("Search"):
    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings)
    
    st.write("### Top 10 Relevant Documents:")
    for doc, score in results:
        st.write(f"- **{doc.strip()}** (Score: {score:.4f})")
