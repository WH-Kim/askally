# modules/tools.py

import streamlit as st
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from .config import DB_PATH

@st.cache_resource
def get_sql_tools(_llm):
    db = SQLDatabase.from_uri(DB_PATH)
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=_llm)
    return sql_toolkit.get_tools()

def create_rag_retriever(_vector_store):
    if _vector_store is None:
        return None
    return _vector_store.as_retriever()