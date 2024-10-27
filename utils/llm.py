import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def GroqChat(input_text):
    """Send a prompt to GroqChat and get a response."""
    llm = Groq(
        model="llama3-70b-8192",
        api_key=GROQ_API_KEY,
    )
    response = llm.complete(input_text)
    return response.text
