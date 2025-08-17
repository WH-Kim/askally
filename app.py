# app.py

import os
import streamlit as st
import pandas as pd
import asyncio
import json
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from modules.graph import create_supervisor_graph
from modules.utils import get_db_schema_and_samples, load_few_shot_examples_from_jsonl, load_or_create_vector_db, get_indexed_doc_samples, save_uploaded_files
from modules.config import DB_PATH, AVAILABLE_OPENAI_MODELS, AVAILABLE_OLLAMA_MODELS, RAG_DOCUMENTS_PATH, VECTOR_DB_PATH

RECURSION_LIMIT = 25

st.set_page_config(page_title="🤖 Dynamic Supervisor Agent", page_icon="🤖", layout="wide")

with st.sidebar:
    st.title("⚙️ 설정")
    model_options = [f"OpenAI: {m}" for m in AVAILABLE_OPENAI_MODELS] + [f"Ollama: {m}" for m in AVAILABLE_OLLAMA_MODELS]
    if 'selected_model_option' not in st.session_state:
        st.session_state.selected_model_option = model_options[0]
    selected_option = st.selectbox("LLM 모델 선택", options=model_options, key="selected_model_option")
    provider, model_name = selected_option.split(": ")
    st.session_state.model_provider = provider
    st.session_state.selected_model = model_name
    st.markdown("---")
    if st.button("대화 초기화 🔄", use_container_width=True, type="primary"):
        st.session_state.clear()
        st.rerun()

st.title("🤖 Dynamic Supervisor Multi-Agent Chat")
st.markdown(f"##### 현재 모델: `{st.session_state.get('model_provider', 'OpenAI')}: {st.session_state.get('selected_model', AVAILABLE_OPENAI_MODELS[0])}`")

st.markdown("##### 💬 대화 모드를 선택하세요")
cols = st.columns(4)
with cols[0]:
    if st.button("🤖 자동 (Supervisor)", use_container_width=True):
        st.session_state.chat_mode = "Supervisor"
        st.toast("자동 모드가 선택되었습니다.")
with cols[1]:
    if st.button("📄 RAG 문서 질문", use_container_width=True):
        st.session_state.chat_mode = "RAGAgent"
        st.toast("RAG 질문 모드가 선택되었습니다.")
with cols[2]:
    if st.button("🗃️ DB 질문", use_container_width=True):
        st.session_state.chat_mode = "SQLAgent"
        st.toast("데이터베이스 질문 모드가 선택되었습니다.")
with cols[3]:
    if st.button("💬 일반 대화", use_container_width=True):
        st.session_state.chat_mode = "DirectResponse"
        st.toast("일반 대화 모드가 선택되었습니다.")

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "Supervisor"
st.info(f"현재 모드: **{st.session_state.chat_mode}**")

with st.expander("📄 RAG 문서 정보 및 관리", expanded=False):
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = load_or_create_vector_db(RAG_DOCUMENTS_PATH, VECTOR_DB_PATH)
    st.subheader("인덱싱된 문서 목록")
    indexed_docs = get_indexed_doc_samples(st.session_state.vector_store)
    if indexed_docs:
        for doc in indexed_docs:
            st.caption(f"- {doc}")
    else:
        st.info(f"`{RAG_DOCUMENTS_PATH}` 폴더가 비어있습니다.")
    st.subheader("새 문서 추가")
    uploaded_files = st.file_uploader("업로드할 PDF 파일을 선택하세요.", type="pdf", accept_multiple_files=True)
    if st.button("선택한 파일 업로드 및 재인덱싱", use_container_width=True):
        if uploaded_files:
            save_uploaded_files(uploaded_files, RAG_DOCUMENTS_PATH)
            if 'vector_store' in st.session_state: del st.session_state.vector_store
            st.success("파일 업로드 및 재인덱싱 완료! 페이지를 새로고침하여 적용하세요.")
            st.rerun()
        else:
            st.warning("먼저 파일을 선택해주세요.")

with st.expander("🗂️ 데이터베이스 정보 확인하기", expanded=False):
    tab1, tab2 = st.tabs(["데이터 샘플", "Few-Shot 예시"])
    with tab1:
        st.subheader("테이블 샘플 데이터")
        db_samples = get_db_schema_and_samples(DB_PATH)
        if db_samples:
            for table, df in db_samples.items():
                st.write(f"**- 테이블: `{table}`**")
                st.dataframe(df, use_container_width=True, height=150)
        else:
            st.warning("데이터베이스 샘플을 불러올 수 없습니다.")
    with tab2:
        st.subheader("질의-쿼리 Few-Shot 예시 (자동 로드)")
        if 'few_shot_examples' not in st.session_state:
            st.session_state.few_shot_examples = load_few_shot_examples_from_jsonl("few_shot_examples.jsonl")
        few_shot_examples = st.session_state.get("few_shot_examples", [])
        if few_shot_examples:
            df_examples = pd.DataFrame(few_shot_examples)
            st.dataframe(df_examples, use_container_width=True)
        else:
            st.warning("`few_shot_examples.jsonl` 파일을 찾을 수 없습니다.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""<style>
    .stChatMessage { border-radius: 10px; padding: 10px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stChatMessage[data-testid="stChatMessage-user"] { background-color: #e1f5fe; }
    .stChatMessage[data-testid="stChatMessage-assistant"] { background-color: #f1f8e9; }
</style>""", unsafe_allow_html=True)

provider = st.session_state.get("model_provider", "OpenAI")
model_name = st.session_state.get("selected_model", AVAILABLE_OPENAI_MODELS[0])

if provider == "Ollama":
    llm = ChatOllama(model=model_name, temperature=0)
else:
    llm = ChatOpenAI(model=model_name, temperature=0, max_retries=3, streaming=True)

graph = create_supervisor_graph(llm)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 위에서 대화 모드를 선택하고 질문해주세요."}]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit-thread-{os.urandom(4).hex()}"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        final_response = None
        status_container = st.expander("에이전트 작업 과정", expanded=True)
        response_container = st.empty()
        config = RunnableConfig(recursion_limit=RECURSION_LIMIT, configurable={"thread_id": st.session_state.thread_id})
        
        conversation_history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in st.session_state.messages]
        chat_mode = st.session_state.get("chat_mode")
        inputs = {"messages": conversation_history}
        
        if chat_mode != "Supervisor":
            inputs["next"] = chat_mode
        
        async def stream_and_display():
            final_answer = None
            status_placeholder = status_container.empty()
            status_updates = []
            
            final_state = await asyncio.to_thread(graph.invoke, inputs, config=config)

            final_messages = final_state.get("messages", [])
            if final_messages:
                final_answer = final_messages[-1].content
            
            with status_container:
                st.markdown("#### 에이전트 작업 요약")
                if "query" in final_state and final_state["query"]:
                     st.markdown(f"**Generated SQL Query 🔎**\n```sql\n{final_state['query']}\n```")
                
                if "result" in final_state and final_state["result"]:
                    try:
                        result_data = json.loads(final_state["result"])
                        if "error" in result_data:
                            st.error(f"**Query Error ❌**\n```\n{result_data['error']}\n```")
                        else:
                            st.markdown("**Query Result 📝**")
                            df = pd.DataFrame(result_data.get("data", []), columns=result_data.get("columns", []))
                            st.dataframe(df, use_container_width=True)
                            if result_data.get("truncated"):
                                st.info(f"결과가 너무 많아 최대 {len(df)}건만 표시합니다.")
                    except (json.JSONDecodeError, TypeError):
                        st.markdown(f"**Query Result 📝**\n```\n{final_state['result']}\n```")

            return final_answer or "답변을 찾지 못했습니다."

        try:
            final_response = asyncio.run(stream_and_display())
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            final_response = "죄송합니다, 처리 중 오류가 발생했습니다."

        response_container.markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})