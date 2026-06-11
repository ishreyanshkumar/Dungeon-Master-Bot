"""
config.py — Shared singletons for LLM and embeddings.
Importing this module anywhere gives the same objects, loaded only once.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Single LLM instance shared across all modules
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.8)

# Single embeddings instance shared across all modules
embs = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
