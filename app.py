################# Required module and Packages ##################

import streamlit as st
from components.ui_elements import get_user_query, show_answer_with_chunks
from components.rag_pipeline import rag_pipeline

################ Main Code ########################################

st.title("RAG-based Question Answering System")
st.write("Enter your query below, and the system will retrieve relevant context and generate an answer.")

# Streamlit form for user input
query = get_user_query()

if st.button("Get Answer"):
    if query:
        with st.spinner("Processing..."):
            # Running the RAG pipeline
            retrieved_chunks, answer = rag_pipeline(query)
        
        # Display the answer and retrieved chunks
        show_answer_with_chunks(retrieved_chunks, answer)
    else:
        st.error("Please enter a query to proceed.")
