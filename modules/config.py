# modules/config.py

import os
from dotenv import load_dotenv
from .utils import check_db_exists

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Final Supervisor Agent v9"

# 선택 가능한 LLM 모델 리스트
AVAILABLE_OPENAI_MODELS = ["gpt-4o", "gpt-3.5-turbo"]
AVAILABLE_OLLAMA_MODELS = ["llama3", "qwen2"] # 사용 가능한 Ollama 모델

RAG_DOCUMENTS_PATH = "rag_documents"
VECTOR_DB_PATH = "vector_db"
DB_FILEPATH = check_db_exists("askally.db")
DB_PATH = f"sqlite:///{DB_FILEPATH}"

