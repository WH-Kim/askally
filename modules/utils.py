# modules/utils.py

import os
import sqlite3
import pandas as pd
import streamlit as st
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, inspect, text
import json
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def check_user_exists(user_id: str) -> bool:
    """tb_bestbanker 테이블에서 사용자가 존재하는지 확인합니다."""
    from .config import DB_FILEPATH

    conn = sqlite3.connect(DB_FILEPATH)
    cursor = conn.cursor()
    try:
        query = "SELECT 1 FROM TB_BESTBANKER WHERE ENO = ?"
        cursor.execute(query, (user_id,))
        return cursor.fetchone() is not None
    except Exception:
        return False
    finally:
        conn.close()

def get_user_info(user_id: str) -> dict:
    """Fetches user information from the TB_BESTBANKER table."""
    from .config import DB_FILEPATH

    conn = sqlite3.connect(DB_FILEPATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        query = "SELECT KOR_NM, ENO FROM TB_BESTBANKER WHERE ENO = ? LIMIT 1"
        cursor.execute(query, (user_id,))
        user_data = cursor.fetchone()
        if user_data:
            return dict(user_data)
        return {}
    except Exception:
        return {}
    finally:
        conn.close()


def get_db_schema_and_samples(
    db_path: str,
    user_id: str = "",
    is_admin: bool = False,
    num_samples: int = 3,
    table_names_to_get: list = None,  # 가져올 테이블 이름 목록
    for_prompt: bool = False,  # 프롬프트용 문자열로 반환할지 여부
):
    """데이터베이스 스키마와 샘플 데이터를 가져옵니다. 일반 사용자는 자신의 데이터만 볼 수 있습니다."""
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        inspector = inspect(engine)
        table_names = table_names_to_get or inspector.get_table_names()
        if not table_names:
            return {}
        all_samples = {}

        # 프롬프트용 문자열을 생성하는 경우
        if for_prompt:
            # 1. 기본 스키마 정보 가져오기 (CREATE TABLE)
            db = SQLDatabase(engine=engine)
            schema_info = db.get_table_info(table_names)
            sample_rows_str = "\n\n/*\n"
        else:
            schema_info = ""
            sample_rows_str = ""

        with engine.connect() as connection:
            for table_name in table_names:
                if table_name.upper() == "TB_BESTBANKER" and not is_admin:
                    query = text(
                        f'SELECT * FROM "{table_name}" WHERE eno = :user_id LIMIT :num_samples'
                    )
                    df = pd.read_sql_query(
                        query,
                        connection,
                        params={"user_id": user_id, "num_samples": num_samples},
                    )
                else:
                    query = text(f'SELECT * FROM "{table_name}" LIMIT :num_samples')
                    df = pd.read_sql_query(
                        query, connection, params={"num_samples": num_samples}
                    )
                if for_prompt:
                    sample_rows_str += f"{num_samples} rows from {table_name} table:\n"
                    sample_rows_str += df.to_string(index=False, na_rep="NULL") + "\n"
                else:
                    all_samples[table_name] = df

        if for_prompt:
            sample_rows_str += "*/"
            return schema_info + sample_rows_str
        return all_samples if not for_prompt else None
    except Exception as e:
        # st.error(f"데이터베이스 연결 오류: {e}") # Streamlit 컨텍스트가 없는 곳에서 호출될 수 있으므로 주석 처리
        return None


@st.cache_data
def load_few_shot_examples_from_jsonl(file_path: str):
    if not os.path.exists(file_path):
        return []

    examples = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if "question" in data and "query" in data:
                        examples.append(data)
    except Exception as e:
        st.error(f"Few-Shot 예시 파일 처리 오류: {e}")
        return []
    return examples


@st.cache_data
def load_db_schema_descriptions(file_path: str = "db_schema_descriptions.json"):
    """데이터베이스 스키마 설명을 JSON 파일에서 로드합니다."""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            descriptions = json.load(f)
        return descriptions
    except Exception as e:
        st.error(f"스키마 설명 파일 로딩 오류: {e}")
        return {}


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
            return FAISS.load_local(
                db_path, embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning(f"벡터DB 로딩 실패: {e}. DB를 재생성합니다.")

    with st.spinner(
        f"'{docs_path}' 폴더의 문서를 인덱싱하여 벡터DB를 생성 중입니다..."
    ):
        if not os.path.exists(docs_path) or not os.listdir(docs_path):
            st.warning(
                f"'{docs_path}' 폴더가 비어있습니다. RAG 에이전트를 사용하려면 PDF 파일을 추가해주세요."
            )
            return None

        try:
            loader = DirectoryLoader(
                docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True
            )
            documents = loader.load()
            if not documents:
                st.warning(f"'{docs_path}' 폴더에서 PDF 문서를 찾지 못했습니다.")
                return None

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            docs = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(docs, embeddings)
            vector_store.save_local(db_path)
            st.success(
                f"벡터DB 생성 완료! 총 {len(docs)}개의 문서 조각이 인덱싱되었습니다."
            )
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
    if hasattr(_vector_store, "docstore") and hasattr(_vector_store.docstore, "_dict"):
        for doc in _vector_store.docstore._dict.values():
            doc_sources.add(os.path.basename(doc.metadata.get("source", "알 수 없음")))
            if len(doc_sources) >= num_samples:
                break
    return list(doc_sources)
@st.cache_data
def get_user_performance_history(user_id: str) -> pd.DataFrame:
    """지정된 사용자의 전체 실적 이력과 함께 전국, 영업본부, 직급 평균 총점을 데이터베이스에서 가져옵니다."""
    from .config import DB_FILEPATH
    
    if not user_id or user_id == 'admin':
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(DB_FILEPATH)
        # CTEs를 사용하여 사용자의 데이터, 각 평균을 계산하고 조인합니다.
        query = """
            WITH UserData AS (
                SELECT BAS_DT, PROV_C, PZCNM, NATL_ORD, PZC_ORD, PROV_ORD, EMP_TOT_SCR
                FROM TB_BESTBANKER
                WHERE ENO = :user_id
            ),
            NatlAvg AS (
                SELECT BAS_DT, AVG(EMP_TOT_SCR) as NATL_AVG
                FROM TB_BESTBANKER
                GROUP BY BAS_DT
            ),
            ProvAvg AS (
                SELECT T.BAS_DT, T.PROV_C, AVG(T.EMP_TOT_SCR) as PROV_AVG
                FROM TB_BESTBANKER T
                JOIN (SELECT DISTINCT BAS_DT, PROV_C FROM UserData) U 
                    ON T.BAS_DT = U.BAS_DT AND T.PROV_C = U.PROV_C
                GROUP BY T.BAS_DT, T.PROV_C
            ),
            PzcAvg AS (
                SELECT T.BAS_DT, T.PZCNM, AVG(T.EMP_TOT_SCR) as PZC_AVG
                FROM TB_BESTBANKER T
                JOIN (SELECT DISTINCT BAS_DT, PZCNM FROM UserData) U 
                    ON T.BAS_DT = U.BAS_DT AND T.PZCNM = U.PZCNM
                GROUP BY T.BAS_DT, T.PZCNM
            )
            SELECT
                U.BAS_DT,
                U.NATL_ORD,
                U.PZC_ORD,
                U.PROV_ORD,
                U.EMP_TOT_SCR,
                NA.NATL_AVG,
                PA.PROV_AVG,
                ZA.PZC_AVG
            FROM UserData U
            LEFT JOIN NatlAvg NA ON U.BAS_DT = NA.BAS_DT
            LEFT JOIN ProvAvg PA ON U.BAS_DT = PA.BAS_DT AND U.PROV_C = PA.PROV_C
            LEFT JOIN PzcAvg ZA ON U.BAS_DT = ZA.BAS_DT AND U.PZCNM = ZA.PZCNM
            ORDER BY U.BAS_DT ASC
        """
        df = pd.read_sql_query(query, conn, params={'user_id': user_id})
        return df
    except Exception as e:
        st.error(f"실적 이력 조회 중 오류 발생: {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals() and conn:
            conn.close()
