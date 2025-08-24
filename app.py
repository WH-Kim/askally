# app.py

import os
import streamlit as st
import pandas as pd
import json
import re
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from modules.graph import create_supervisor_graph
from modules.utils import get_db_schema_and_samples, load_few_shot_examples_from_jsonl, load_or_create_vector_db, get_indexed_doc_samples, save_uploaded_files
from modules.config import DB_PATH, AVAILABLE_OPENAI_MODELS, AVAILABLE_OLLAMA_MODELS, RAG_DOCUMENTS_PATH, VECTOR_DB_PATH

RECURSION_LIMIT = 50

st.set_page_config(page_title="🤖 Dynamic Supervisor Agent", page_icon="🤖", layout="wide")

# --- Sidebar ---
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

# --- Main Page ---
st.title("🤖 Dynamic Supervisor Multi-Agent Chat")
st.markdown(f"##### 현재 모델: `{st.session_state.get('model_provider', 'OpenAI')}: {st.session_state.get('selected_model', AVAILABLE_OPENAI_MODELS[0])}`")

st.markdown("##### 💬 대화 모드를 선택하세요")
cols = st.columns(6)
with cols[0]:
    if st.button("🤖 자동 (Supervisor)", use_container_width=True): st.session_state.chat_mode = "Supervisor"
with cols[1]:
    if st.button("📄 RAG", use_container_width=True): st.session_state.chat_mode = "RAGAgent"
with cols[2]:
    if st.button("🗃️ DB 질문", use_container_width=True): st.session_state.chat_mode = "SQLAgent"
with cols[3]:
    if st.button("📊 차트 생성", use_container_width=True): st.session_state.chat_mode = "ChartGeneratorAgent"
with cols[4]:
    if st.button("📄 PDF 보고서", use_container_width=True): st.session_state.chat_mode = "ReportAgent"
with cols[5]:
    if st.button("💬 일반 대화", use_container_width=True): st.session_state.chat_mode = "DirectResponse"


if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "Supervisor"
st.info(f"현재 모드: **{st.session_state.chat_mode}**")


# --- 데이터 로드 및 준비 ---
@st.cache_resource
def load_db_info():
    db_samples = get_db_schema_and_samples(DB_PATH)
    if not db_samples: return None
    # few_shot_examples.jsonl 파일이 없어도 오류가 발생하지 않도록 처리
    try:
        few_shot_examples = load_few_shot_examples_from_jsonl("few_shot_examples.jsonl")
    except FileNotFoundError:
        few_shot_examples = []
    return {"few_shot_examples": few_shot_examples}

db_info = load_db_info()

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = load_or_create_vector_db(RAG_DOCUMENTS_PATH, VECTOR_DB_PATH)
retriever = st.session_state.vector_store.as_retriever() if st.session_state.vector_store else None


with st.expander("📄 RAG 문서 정보 및 관리", expanded=False):
    st.subheader("인덱싱된 문서 목록")
    if st.session_state.vector_store:
        indexed_docs = get_indexed_doc_samples(st.session_state.vector_store)
        if indexed_docs:
            for doc in indexed_docs: st.caption(f"- {doc}")
        else:
            st.info("인덱싱된 문서가 없습니다.")
    else:
        st.info(f"`{RAG_DOCUMENTS_PATH}` 폴더가 비어있습니다.")

    st.subheader("새 문서 추가")
    uploaded_files = st.file_uploader("업로드할 PDF 파일을 선택하세요.", type="pdf", accept_multiple_files=True)
    if st.button("선택한 파일 업로드 및 재인덱싱", use_container_width=True):
        if uploaded_files:
            save_uploaded_files(uploaded_files, RAG_DOCUMENTS_PATH)
            if 'vector_store' in st.session_state:
                del st.session_state.vector_store
            st.rerun()
        else:
            st.warning("먼저 파일을 선택해주세요.")

with st.expander("🗂️ 데이터베이스 정보 확인하기", expanded=False):
    db_samples = get_db_schema_and_samples(DB_PATH)
    if db_samples:
        for table, df in db_samples.items():
            st.write(f"**- 테이블: `{table}`**")
            st.dataframe(df, use_container_width=True, height=150)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""<style>
    .stChatMessage { border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    .stChatMessage[data-testid="stChatMessage-user"] { background-color: #e1f5fe; }
    .stChatMessage[data-testid="stChatMessage-assistant"] { background-color: #f1f8e9; }
</style>""", unsafe_allow_html=True)


# --- LLM 및 그래프 초기화 ---
provider = st.session_state.get("model_provider", "OpenAI")
model_name = st.session_state.get("selected_model", AVAILABLE_OPENAI_MODELS[0])
if provider == "Ollama":
    llm = ChatOllama(model=model_name, temperature=0)
else:
    llm = ChatOpenAI(model=model_name, temperature=0, max_retries=3, streaming=True)

graph = create_supervisor_graph(llm, retriever, db_info)


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 위에서 대화 모드를 선택하고 질문해주세요."}]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit-thread-{os.urandom(4).hex()}"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg.get("content", "")
        image_paths = re.findall(r'[\w-]+\.png', content)
        report_paths = re.findall(r'[\w-]+\.pdf', content)
        
        text_content = content
        for path in image_paths + report_paths:
            text_content = text_content.replace(path, "").strip()

        if text_content:
            st.write(text_content)
        
        for path in image_paths:
            if os.path.exists(path):
                st.image(path)
        
        for path in report_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    st.download_button(
                        label=f"📄 {os.path.basename(path)} 다운로드",
                        data=f,
                        file_name=os.path.basename(path),
                        mime="application/pdf"
                    )

if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("에이전트가 생각 중입니다..."):
            response_container = st.empty()
            
            config = RunnableConfig(recursion_limit=RECURSION_LIMIT, configurable={"thread_id": st.session_state.thread_id})
            
            # 대화 기록을 LangChain 메시지 형식으로 변환
            conversation_history = [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" 
                else AIMessage(content=msg["content"]) 
                for msg in st.session_state.messages
            ]

            chat_mode = st.session_state.get("chat_mode")
            inputs = {"messages": conversation_history}
            if chat_mode != "Supervisor":
                inputs["next"] = chat_mode
            
            full_response = ""
            node_statuses = {}
            active_node_name = None
            final_state = None

            try:
                for chunk in graph.stream(inputs, config=config, stream_mode="updates"):
                    node_name = list(chunk.keys())[0]

                    if node_name == "__end__":
                        final_state = chunk[node_name]
                        break

                    if active_node_name and active_node_name in node_statuses:
                        node_statuses[active_node_name].update(state="complete", expanded=False)

                    active_node_name = node_name
                    node_update = chunk[node_name]
                    
                    if node_name not in node_statuses:
                        node_statuses[node_name] = st.status(f"**실행 중:** `{node_name}`", state="running", expanded=True)

                    with node_statuses[node_name]:
                        if node_name == "Supervisor":
                            if next_agent := node_update.get("next"): st.markdown(f"↪️ 다음 작업으로 **`{next_agent}`** 호출")
                        elif node_name == "select_tables": st.markdown(f"**- 관련된 테이블 선택:** `{node_update.get('table_names')}`")
                        elif node_name == "get_schema": st.markdown(f"**- 테이블 스키마 조회**")
                        elif node_name == "query_gen": 
                            st.markdown(f"**- SQL 쿼리 생성 중...**")
                            # [수정] query_gen 노드에서 생성된 SQL 쿼리를 화면에 표시
                            if query := node_update.get('query'):
                                 st.markdown(f"**- 생성된 SQL 쿼리:**")
                                 st.code(query, language='sql')
                        
                        if "result" in node_update:
                            st.markdown("**- 실행 결과:**")
                            try:
                                result_data = json.loads(node_update["result"])
                                if "data" in result_data:
                                    df = pd.DataFrame(result_data.get("data", []), columns=result_data.get("columns", []))
                                    st.dataframe(df)
                                elif "error" in result_data: st.error(result_data['error'])
                            except (json.JSONDecodeError, TypeError): st.text(node_update["result"])

                    if "messages" in node_update:
                        last_message = node_update["messages"][-1]
                        if isinstance(last_message, AIMessageChunk):
                            full_response += last_message.content
                            response_container.markdown(full_response + "▌")

                final_answer = ""
                # 최종 답변은 마지막 __end__ 상태의 메시지에서 가져옴
                if final_state and final_state.get("messages"):
                    final_answer = final_state.get("messages", [])[-1].content
                elif full_response:
                    final_answer = full_response

                if final_answer:
                    response_container.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
                # PDF 생성이 완료되면 rerun하여 다운로드 버튼을 즉시 표시
                if final_answer and re.search(r'[\w-]+\.pdf', final_answer):
                    st.rerun()

            except Exception as e:
                error_message = f"죄송합니다, 처리 중 오류가 발생했습니다: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})