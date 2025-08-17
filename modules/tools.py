# modules/tools.py

import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from .config import DB_PATH
from typing import List
import sqlite3
import json

MAX_ROWS_TO_DISPLAY = 20

@st.cache_resource
def get_db():
    return SQLDatabase.from_uri(DB_PATH)

@tool
def list_tables() -> List[str]:
    """데이터베이스에 있는 테이블 목록을 반환합니다."""
    db = get_db()
    return db.get_table_names()

@tool
def get_schema(table_names: str) -> str:
    """주어진 테이블에 대한 스키마와 샘플 행을 반환합니다."""
    db = get_db()
    return db.get_table_info(table_names.split(","))

@tool
def execute_query(query: str) -> str:
    """
    주어진 SQL 쿼리를 실행하고 결과를 JSON 형식으로 반환합니다.
    결과는 컬럼과 데이터 리스트를 포함하며, 건수가 많을 경우 일부만 반환합니다.
    """
    db_path = DB_PATH.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description] if cursor.description else []
        
        truncated = False
        if "limit" not in query.lower() and len(rows) > MAX_ROWS_TO_DISPLAY:
            rows = rows[:MAX_ROWS_TO_DISPLAY]
            truncated = True
            
        result = { "columns": columns, "data": rows, "truncated": truncated }
    except Exception as e:
        result = {"error": str(e)}
    finally:
        conn.close()
        
    return json.dumps(result, ensure_ascii=False)

def get_rag_tool():
    vector_store = st.session_state.get("vector_store")
    if not vector_store:
        return None
    retriever = vector_store.as_retriever()
    return create_retriever_tool(
        retriever,
        "pdf_document_retriever",
        "Searches and returns information from the user's PDF documents.",
    )

def get_sql_tools():
    return [list_tables, get_schema, execute_query]