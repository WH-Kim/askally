# modules/config.py

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Final Supervisor Agent v9"

# 선택 가능한 LLM 모델 리스트
AVAILABLE_OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

RAG_DOCUMENTS_PATH = "rag_documents"
VECTOR_DB_PATH = "vector_db"
DB_FILEPATH = "askally.db"
DB_PATH = f"sqlite:///{DB_FILEPATH}"
