# tercoReader

**tercoReader** is a RAG (Retrieval-Augmented Generation) based question-answering system built with Streamlit. It utilizes advanced text embedding techniques, a Groq LLM API, and a PostgreSQL database to retrieve relevant document chunks for generating precise responses to user queries.

## Table of Contents
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)

---

## Tech Stack

**Frontend:**
- **Streamlit**: Streamlit provides a user-friendly interface for inputting questions and viewing results. It’s a popular open-source app framework for creating machine learning and data science applications with minimal effort.

**Backend:**
- **Python**: The core language for the project, handling data processing, model interaction, and logic.
- **SQLAlchemy**: Used as the ORM (Object Relational Mapping) to manage interactions with the PostgreSQL database, creating and managing tables for documents and embeddings.


**Machine Learning & NLP:**
- **Hugging Face Transformers (BERT)**: BERT is used to generate embeddings for chunks and query text. These embeddings allow the system to determine the similarity between query and document content.
- **GroqChat API**: This API generates responses based on retrieved context, providing high-quality LLM responses. This API offers an efficient way to generate responses based on custom contexts.

**Database:**
- **PostgreSQL**: This relational database stores document chunks and their embeddings. PostgreSQL is highly reliable, making it well-suited for managing and querying document embeddings.


---

## Setup and Installation
1. Clone the repo
```bash
git clone https://github.com/Ranjan00001/tercoReader.git
```
2. Install requirements
```bash
pip install -r requirements.txt
```
3. Run the code
```bash
streamlit run app.py
``` 