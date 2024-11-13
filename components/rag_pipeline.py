from utils.helpers import get_bert_embeddings
from rank_bm25 import BM25Okapi
from components.database import Session, ChunkEmbedding
import numpy as np
from utils.llm import GroqChat

def get_bm25_top_k(query, chunks, k=100):
    """Use BM25 to retrieve the top k relevant chunks based on the query."""
    # Tokenize chunks for BM25 processing
    tokenized_chunks = [chunk.split(" ") for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    query_tokens = query.split(" ")
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Get the indices of the top k BM25-scored chunks
    top_k_indices = np.argsort(bm25_scores)[-k:][::-1]
    return top_k_indices

def get_top_10_chunks(query):
    """Retrieve the top 10 most relevant chunks using BM25 + Embedding strategy."""
    query_vector = np.array(get_bert_embeddings(query))
    
    # Initialize session and fetch all chunks and embeddings
    session = Session()
    results = session.query(ChunkEmbedding.chunk, ChunkEmbedding.embedding).all()
    session.close()
    
    if not results:
        return []
    
    # Separate chunks and embeddings from query results
    chunks = [result[0] for result in results]
    embeddings = np.array([result[1] for result in results])

    # Step 1: Use BM25 to retrieve top 100 chunks based on initial relevance
    top_k_bm25_indices = get_bm25_top_k(query, chunks, k=100)
    
    # Filter to the top 100 BM25 chunks and their embeddings
    bm25_chunks = [chunks[i] for i in top_k_bm25_indices]
    bm25_embeddings = embeddings[top_k_bm25_indices]

    # Step 2: Use dot product on the BM25-filtered embeddings to find the top 10
    similarities = np.dot(bm25_embeddings, query_vector.T).flatten()
    top10_indices = similarities.argsort()[::-1][:10]
    top_10_chunks = [bm25_chunks[i] for i in top10_indices]

    return top_10_chunks

def rag_pipeline(query):
    """Main RAG pipeline to retrieve top chunks and generate a response."""
    retrieved_chunks = get_top_10_chunks(query)
    
    prompt = (
        'You are a helpful and informative bot that answers questions using text from the reference passage included below.' 
        'Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. '
        'However, you are talking to a non-technical audience, so be sure to break down complicated concepts and '
        'strike a friendly and converstional tone. If the passage is irrelevant to the answer, you may ignore it.'
        f'QUESTION: {query}   PASSAGE: {retrieved_chunks}'
    )
    
    response = GroqChat(prompt)
    return retrieved_chunks, response
