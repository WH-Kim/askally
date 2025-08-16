# modules/utils.py

import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect
import json
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def check_db_exists(db_filepath: str):
    if not os.path.exists(db_filepath):
        st.error(f"데이터베이스 파일('{db_filepath}')을 찾을 수 없습니다.")
        st.stop()
    return db_filepath

@st.cache_data(ttl=600)
def get_db_schema_and_samples(db_path: str, num_samples: int = 3):
    try:
        engine = create_engine(db_path)
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        if not table_names:
            st.warning("데이터베이스에서 테이블을 찾을 수 없습니다.")
            return {}
        all_samples = {}
        with engine.connect() as connection:
            for table_name in table_names:
                query = f'SELECT * FROM "{table_name}" LIMIT {num_samples}'
                df = pd.read_sql_query(query, connection)
                all_samples[table_name] = df
        return all_samples
    except Exception as e:
        st.error(f"데이터베이스 연결 오류: {e}")
        return None

@st.cache_data
def load_few_shot_examples_from_jsonl(file_path: str):
    if not os.path.exists(file_path):
        return []
    
    examples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if 'question' in data and 'query' in data:
                        examples.append(data)
    except Exception as e:
        st.error(f"Few-Shot 예시 파일 처리 오류: {e}")
        return []
    return examples

# --- 신규/수정된 RAG 관련 함수 ---
def save_uploaded_files(uploaded_files, directory):
    """업로드된 파일들을 지정된 디렉토리에 저장합니다."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"{len(uploaded_files)}개의 파일이 성공적으로 업로드되었습니다.")

# st.cache_resource를 제거하여 파일 추가 시 재생성되도록 변경
def load_or_create_vector_db(docs_path: str, db_path: str, force_recreate=False):
    """
    벡터DB를 로드하거나 새로 생성합니다. force_recreate가 True이면 항상 재생성합니다.
    """
    faiss_index_path = os.path.join(db_path, "index.faiss")
    
    if os.path.exists(faiss_index_path) and not force_recreate:
        st.info("기존 벡터DB를 로드합니다.")
        embeddings = OpenAIEmbeddings()
        try:
            return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"벡터DB 로딩 실패: {e}. DB를 재생성합니다.")

    with st.spinner(f"'{docs_path}' 폴더의 문서를 인덱싱하여 벡터DB를 생성 중입니다..."):
        if not os.path.exists(docs_path) or not os.listdir(docs_path):
            st.warning(f"'{docs_path}' 폴더가 비어있습니다. RAG 에이전트를 사용하려면 PDF 파일을 추가해주세요.")
            return None
            
        try:
            loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True)
            documents = loader.load()
            if not documents:
                st.warning(f"'{docs_path}' 폴더에서 PDF 문서를 찾지 못했습니다.")
                return None

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(docs, embeddings)
            vector_store.save_local(db_path)
            st.success(f"벡터DB 생성 완료! 총 {len(docs)}개의 문서 조각이 인덱싱되었습니다.")
            return vector_store
        except Exception as e:
            st.error(f"벡터DB 생성 중 오류 발생: {e}")
            return None

@st.cache_data
def get_indexed_doc_samples(_vector_store, num_samples: int = 5):
    """인덱싱된 벡터 저장소에서 문서 샘플(파일명)을 반환합니다."""
    if _vector_store is None:
        return []
    
    doc_sources = set()
    if hasattr(_vector_store, 'docstore') and hasattr(_vector_store.docstore, '_dict'):
        for doc in _vector_store.docstore._dict.values():
            doc_sources.add(os.path.basename(doc.metadata.get('source', '알 수 없음')))
            if len(doc_sources) >= num_samples:
                break
    return list(doc_sources)