from utils.helpers import get_bert_embeddings
from utils.llm import GroqChat
from components.database import Session, ChunkEmbedding
import numpy as np

def get_top_10_chunks(query):
    """Retrieve the top 10 most relevant chunks based on the query."""
    query_vector = np.array(get_bert_embeddings(query))
    session = Session()
    results = session.query(ChunkEmbedding.chunk, ChunkEmbedding.embedding).all()
    session.close()
    
    if not results:
        return []

    chunks = [result[0] for result in results]
    embeddings = np.array([result[1] for result in results])

    similarities = np.dot(embeddings, query_vector.T).flatten()
    top10 = similarities.argsort()[::-1][:10]
    top_10_chunks = [chunks[i] for i in top10]

    return top_10_chunks

def rag_pipeline(query):
    """Main RAG pipeline to retrieve top chunks and generate a response."""
    retrieved_chunks = get_top_10_chunks(query)
    
    prompt = (
        f"Please answer the following query: {query}\n\n"
        "Here are the top retrieved contexts related to the user query. Please focus strictly on these contexts when answering the query:\n\n"
        + "\n".join(retrieved_chunks)
    )
    
    response = GroqChat(prompt)
    return retrieved_chunks, response
