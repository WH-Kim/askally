# app.py

import os
import streamlit as st
import pandas as pd
import asyncio
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langgraph.graph import END

from modules.graph import create_supervisor_graph
from modules.utils import get_db_schema_and_samples, load_few_shot_examples_from_jsonl, load_or_create_vector_db, get_indexed_doc_samples, save_uploaded_files
from modules.config import DB_PATH, AVAILABLE_OPENAI_MODELS, AVAILABLE_OLLAMA_MODELS, RAG_DOCUMENTS_PATH, VECTOR_DB_PATH

RECURSION_LIMIT = 15

# --- Streamlit UI 구성 ---
st.set_page_config(page_title="🤖 Dynamic Supervisor Agent", page_icon="🤖", layout="wide")

# --- 사이드바 ---
with st.sidebar:
    st.title("⚙️ 설정")
    
    # 1. LLM 모델 선택 (통합 메뉴)
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

# --- 메인 화면 ---
st.title("🤖 Dynamic Supervisor Multi-Agent Chat")
st.markdown(f"##### 현재 모델: `{st.session_state.get('model_provider', 'OpenAI')}: {st.session_state.get('selected_model', AVAILABLE_OPENAI_MODELS[0])}`")

cols = st.columns(3)
with cols[0]:
    if st.button("💬 일반 대화", use_container_width=True):
        st.session_state.chat_mode = "DirectResponse"
        st.toast("일반 대화 모드가 선택되었습니다.")
with cols[1]:
    if st.button("📄 RAG 문서에 질문하기", use_container_width=True):
        st.session_state.chat_mode = "PDF_RAG_Agent"
        st.toast("RAG 질문 모드가 선택되었습니다.")
with cols[2]:
    if st.button("🗃️ 데이터베이스에 질문하기", use_container_width=True):
        st.session_state.chat_mode = "SQLAgent"
        st.toast("데이터베이스 질문 모드가 선택되었습니다.")

# --- 작업 모드별 정보 표시 ---
if st.session_state.get("chat_mode") == "PDF_RAG_Agent":
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
                # 벡터 DB를 강제로 재생성하도록 세션 상태를 삭제
                if 'vector_store' in st.session_state:
                    del st.session_state.vector_store
                st.success("파일 업로드 및 재인덱싱 완료! 페이지를 새로고침하여 적용하세요.")
                st.rerun()
            else:
                st.warning("먼저 파일을 선택해주세요.")

if st.session_state.get("chat_mode") == "SQLAgent":
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

# --- 그래프 및 세션 초기화 ---
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

if prompt := st.chat_input("작업 모드를 선택하고 질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        final_response = None
        
        # 에이전트 작업 과정과 최종 답변을 분리하여 표시
        status_container = st.expander("에이전트 작업 과정", expanded=True)
        response_container = st.empty()
        
        config = RunnableConfig(recursion_limit=RECURSION_LIMIT, configurable={"thread_id": st.session_state.thread_id})
        
        chat_mode = st.session_state.get("chat_mode", None)
        if not chat_mode:
            final_response = "먼저 상단의 대화 모드를 선택해주세요."
        else:
            current_prompt = prompt
            if chat_mode == "PDF_RAG_Agent":
                current_prompt = f"폴더에 있는 PDF 문서를 참고하여 다음 질문에 답해주세요: {prompt}"
            elif chat_mode == "SQLAgent":
                current_prompt = f"데이터베이스를 조회하여 다음 질문에 답해주세요: {prompt}"

            inputs = {"messages": st.session_state.messages} # 전체 대화 기록 전달
            
            async def stream_and_display():
                full_response = ""
                status_updates = []
                
                async for event in graph.astream_events(inputs, config=config, version="v1"):
                    kind = event["event"]
                    name = event["name"]
                    
                    if kind == "on_chain_end":
                        if name == "Supervisor":
                            if event["data"]["output"]:
                                status_updates.append(f"**Supervisor:** 다음 작업자로 `{event['data']['output'].next}`를 선택했습니다.")
                        elif name in ["SQLAgent", "PDF_RAG_Agent", "DirectResponse"]:
                             status_updates.append(f"**{name}:** 작업을 시작합니다...")
                    
                    elif kind == "on_tool_start":
                        status_updates.append(f"**{name} (도구 사용 🛠️):** `{event['name']}` 실행 중...")
                        if event['name'] == 'sql_db_query':
                            status_updates.append(f"```sql\n{event['data']['input']['query']}\n```")
                    
                    elif kind == "on_tool_end":
                        status_updates.append(f"**{name} (결과 📝):** `{event['name']}` 실행 완료.")
                    
                    elif kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if isinstance(chunk, AIMessage) and chunk.content:
                            full_response += chunk.content
                            response_container.markdown(full_response + "▌")
                
                return full_response, status_updates

            try:
                # 비동기 함수 실행
                final_response, status_history = asyncio.run(stream_and_display())
                
                # 최종 상태 업데이트
                with status_container:
                    for update in status_history:
                        st.markdown(update)

            except Exception as e:
                if "recursion limit" in str(e).lower():
                    final_response = f"죄송합니다. 에이전트가 {RECURSION_LIMIT}번의 시도 후에도 답변을 찾지 못했습니다."
                else:
                    final_response = f"죄송합니다, 오류가 발생했습니다: {e}"
                st.error(final_response)

        if final_response:
            response_container.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
        else:
            fallback_message = "죄송합니다. 답변을 생성하지 못했습니다."
            st.error(fallback_message)
            if st.session_state.messages and st.session_state.messages[-1]['role'] == 'user':
                st.session_state.messages.pop()