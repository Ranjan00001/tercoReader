import re
import torch
from transformers import BertTokenizer, BertModel

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased')

def clean_text(text):
    """Clean text by removing unwanted characters and formatting."""
    text = re.sub(r'(?i)copyright.*', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_bert_embeddings(text):
    """Compute BERT embeddings for the given text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        embeddings = torch.mean(last_hidden_state, dim=1)
    return embeddings.squeeze().tolist()
