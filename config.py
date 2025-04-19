import os

from dotenv import load_dotenv

load_dotenv(verbose=True, override=True)

PDF_PATH = os.getenv("PDF_PATH")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", 'deepseek-r1:8b')
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", 'llama3.2')
VECTOR_STORE_ROOT = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
