import streamlit as st

def get_user_query():
    """Get the user's query input from Streamlit."""
    return st.text_input("Enter your question here:", key="user_query_input")

def show_answer_with_chunks(retrieved_chunks, answer):
    """Display the answer and retrieved chunks with an expander for chunks."""
    st.subheader("Top Retrieved Chunks:")
    with st.expander("View Retrieved Chunks"):
        for i, chunk in enumerate(retrieved_chunks, 1):
            st.markdown(f"**Chunk {i}:** {chunk}")
    
    st.subheader("Generated Answer:")
    st.write(answer)
