# app.py

import os
import streamlit as st
import asyncio
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from modules.graph import create_supervisor_graph
from modules.utils import (
    get_db_schema_and_samples,
    check_user_exists,
    load_few_shot_examples_from_jsonl,
    load_or_create_vector_db,
    get_indexed_doc_samples,
    load_db_schema_descriptions,
    save_uploaded_files,
    get_user_performance_history,
)
from modules.config import (
    DB_PATH,
    AVAILABLE_OPENAI_MODELS,
    RAG_DOCUMENTS_PATH,
    VECTOR_DB_PATH,
    DB_FILEPATH,
)

RECURSION_LIMIT = 25

if not os.path.exists(DB_FILEPATH):
    st.error(
        f"데이터베이스 파일('{DB_FILEPATH}')을 찾을 수 없습니다. `askally.db` 파일이 프로젝트 루트에 있는지 확인해주세요."
    )
    st.stop()


def login_page():
    """로그인 페이지를 표시합니다."""
    st.title("🤖 AskAlly")
    with st.form("login_form"):
        employee_id = st.text_input("사번을 입력하세요.")
        submitted = st.form_submit_button("로그인")
        if submitted:
            if employee_id.lower() == "admin":
                st.session_state.logged_in = True
                st.session_state.user_id = "admin"
                st.session_state.is_admin = True
                st.rerun()
            elif check_user_exists(user_id=employee_id):
                st.session_state.logged_in = True
                st.session_state.user_id = employee_id
                st.session_state.is_admin = False
                st.rerun()
            else:
                st.error("존재하지 않는 사번입니다. 다시 시도해주세요.")


if st.session_state.get("logged_in", False) is False:
    login_page()
    st.stop()

st.set_page_config(
    page_title="AskAlly", page_icon="🤖", layout="wide"
)

with st.sidebar:
    st.success(f"**{st.session_state.user_id}**님, 환영합니다.")
    if st.button("로그아웃 🚪", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key != "selected_model_option":
                del st.session_state[key]
        st.rerun()
    st.markdown("---")
    st.title("⚙️ 설정")
    model_options = [f"OpenAI: {m}" for m in AVAILABLE_OPENAI_MODELS]
    if "selected_model_option" not in st.session_state:
        st.session_state.selected_model_option = model_options[0]
    selected_option = st.selectbox(
        "LLM 모델 선택", options=model_options, key="selected_model_option"
    )
    provider, model_name = selected_option.split(": ")
    st.session_state.model_provider = provider
    st.session_state.selected_model = model_name
    st.markdown("---")
    if st.button("대화 초기화 🔄", use_container_width=True, type="primary"):
        # st.session_state.clear()는 로그인 정보까지 삭제하므로, 대화 관련 세션만 초기화합니다.
        keys_to_clear = ["messages", "thread_id", "last_sql_result"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

st.markdown(
    f"##### 현재 모델: `{st.session_state.get('model_provider', 'OpenAI')}: {st.session_state.get('selected_model', AVAILABLE_OPENAI_MODELS[0])}`"
)

st.markdown("##### 💬 대화 모드를 선택하세요")
cols = st.columns(4)
with cols[0]:
    if st.button("🤖 자동 (Supervisor)", use_container_width=True):
        st.session_state.chat_mode = "Supervisor"
        st.toast("자동 모드가 선택되었습니다.")
with cols[1]:
    if st.button("📋 보고서 생성", use_container_width=True):
        st.session_state.chat_mode = "ReportAgent"
        st.toast(
            "보고서 생성 모드가 선택되었습니다. '지금까지 대화 요약해줘'와 같이 요청해보세요."
        )
with cols[2]:
    if st.button("📄 RAG 문서 질문", use_container_width=True):
        st.session_state.chat_mode = "RAGAgent"
        st.toast("RAG 질문 모드가 선택되었습니다.")
with cols[3]:
    if st.button("🗃️ DB SQL 질문", use_container_width=True):
        st.session_state.chat_mode = "SQLAgent"
        st.toast("데이터베이스 질문 모드가 선택되었습니다.")


if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "Supervisor"
st.info(f"현재 모드: **{st.session_state.chat_mode}**")

# --- 실적 추이 시각화 (개선된 버전) --- #
with st.expander("📈 내 실적 변화 추이 보기", expanded=False):
    perf_history_df = get_user_performance_history(st.session_state.user_id)
    if not perf_history_df.empty:
        # --- Font Setup ---
        font_path = "fonts/NanumGothic.ttf"
        font_prop = None
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
        else:
            st.warning(f"폰트 파일이 '{font_path}'에 없어 한글이 깨질 수 있습니다.")

        plt.rcParams["axes.unicode_minus"] = False
        sns.set_theme(style="whitegrid", palette="pastel")

        # --- Data Preparation ---
        try:
            perf_history_df["BAS_DT"] = pd.to_datetime(perf_history_df["BAS_DT"], format="%Y%m%d")
            perf_history_df = perf_history_df.sort_values(by="BAS_DT")
        except Exception as e:
            st.error(f"데이터의 날짜 형식에 문제가 있어 차트를 그릴 수 없습니다: {e}")
            st.stop()

        col1, col2 = st.columns(2)
        x_min = perf_history_df["BAS_DT"].min()
        x_max = perf_history_df["BAS_DT"].max()

        with col1:
            st.subheader("🏆 순위 변화")
            rank_df = perf_history_df.set_index("BAS_DT")
            rank_cols = {"NATL_ORD": "전국순위", "PZC_ORD": "직급순위", "PROV_ORD": "본부순위"}
            rank_df = rank_df[list(rank_cols.keys())].copy().rename(columns=rank_cols)

            fig_rank, ax_rank = plt.subplots(figsize=(6, 4))
            for col in rank_df.columns:
                ax_rank.plot(rank_df.index, rank_df[col], marker='o', linestyle='-', label=col)
            
            ax_rank.invert_yaxis()
            ax_rank.set_title("월별 순위 변화", fontproperties=font_prop, fontsize=12)
            ax_rank.set_ylabel("순위", fontproperties=font_prop)
            ax_rank.legend(prop=font_prop, fontsize=8)
            ax_rank.set_xlim(x_min, x_max)
            plt.setp(ax_rank.get_xticklabels(), rotation=45, ha="right")
            
            plt.tight_layout()
            st.pyplot(fig_rank)
            plt.close(fig_rank)

        with col2:
            st.subheader("💯 총점 변화")
            score_df = perf_history_df.set_index("BAS_DT")
            score_cols_map = {
                "EMP_TOT_SCR": "내 총점", "NATL_AVG": "전국 평균",
                "PROV_AVG": "영업본부 평균", "PZC_AVG": "직급 평균"
            }
            
            if all(col in score_df.columns for col in score_cols_map.keys()):
                score_df = score_df[list(score_cols_map.keys())].copy().rename(columns=score_cols_map)
                fig_score, ax_score = plt.subplots(figsize=(6, 4))
                for col in score_df.columns:
                    ax_score.plot(score_df.index, score_df[col], marker='o', linestyle='-' if '평균' not in col else '--', label=col)

                ax_score.set_title("월별 총점 변화", fontproperties=font_prop, fontsize=12)
                ax_score.set_ylabel("점수", fontproperties=font_prop)
                ax_score.legend(prop=font_prop, fontsize=8)
                ax_score.set_xlim(x_min, x_max)
                plt.setp(ax_score.get_xticklabels(), rotation=45, ha="right")

                plt.tight_layout()
                st.pyplot(fig_score)
                plt.close(fig_score)
            else:
                st.warning("총점 평균 데이터가 일부 존재하지 않아 차트를 표시할 수 없습니다.")

with st.expander("📄 RAG 문서 정보 및 관리", expanded=False):
    force_recreate = st.session_state.get("force_rag_recreate", False)
    if "vector_store" not in st.session_state or force_recreate:
        st.session_state.vector_store = load_or_create_vector_db(
            RAG_DOCUMENTS_PATH, VECTOR_DB_PATH, force_recreate=force_recreate
        )
        if force_recreate:
            del st.session_state.force_rag_recreate

    st.subheader("인덱싱된 문서 목록")
    indexed_docs = get_indexed_doc_samples(st.session_state.vector_store)
    if indexed_docs:
        for doc in indexed_docs:
            st.caption(f"- {doc}")
    else:
        st.info(f"`{RAG_DOCUMENTS_PATH}` 폴더가 비어있습니다.")
    st.subheader("새 문서 추가")
    uploaded_files = st.file_uploader(
        "업로드할 PDF 파일을 선택하세요.", type="pdf", accept_multiple_files=True
    )
    if st.button("선택한 파일 업로드 및 재인덱싱", use_container_width=True):
        if uploaded_files:
            save_uploaded_files(uploaded_files, RAG_DOCUMENTS_PATH)
            if "vector_store" in st.session_state:
                del st.session_state.vector_store
            st.session_state.force_rag_recreate = True
            st.success("파일 업로드 및 재인덱싱 완료! 페이지를 새로고침하면 적용됩니다.")
            st.rerun()
        else:
            st.warning("먼저 파일을 선택해주세요.")

with st.expander("🗂️ 데이터베이스 정보 확인하기", expanded=False):
    tab1, tab2, tab3 = st.tabs(["데이터 샘플", "Few-Shot 예시", "테이블/컬럼 설명"])
    with tab1:
        st.subheader("테이블 샘플 데이터")
        db_samples = get_db_schema_and_samples(
            db_path=DB_FILEPATH,
            user_id=st.session_state.user_id,
            is_admin=st.session_state.is_admin,
        )
        if db_samples:
            for table, df in db_samples.items():
                st.write(f"**- 테이블: `{table}`**")
                st.dataframe(df, use_container_width=True, height=150)
        else:
            st.warning("데이터베이스 샘플을 불러올 수 없습니다.")
    with tab2:
        st.subheader("질의-쿼리 Few-Shot 예시 (자동 로드)")
        if "few_shot_examples" not in st.session_state:
            st.session_state.few_shot_examples = load_few_shot_examples_from_jsonl(
                "few_shot_examples.jsonl"
            )
        few_shot_examples = st.session_state.get("few_shot_examples", [])
        if few_shot_examples:
            df_examples = pd.DataFrame(few_shot_examples)
            st.dataframe(df_examples, use_container_width=True)
        else:
            st.warning("`few_shot_examples.jsonl` 파일을 찾을 수 없습니다.")
    with tab3:
        st.subheader("테이블/컬럼 한국어 설명 (db_schema_descriptions.json)")
        if "db_descriptions" not in st.session_state:
            st.session_state.db_descriptions = load_db_schema_descriptions()

        db_descriptions = st.session_state.get("db_descriptions", {})
        if db_descriptions:
            st.info("테이블 및 컬럼 설명을 성공적으로 로드했습니다.")
            st.json(db_descriptions)
        else:
            st.warning(
                "`db_schema_descriptions.json` 파일을 찾을 수 없거나 내용이 비어있습니다. 파일을 생성하면 SQL 생성 정확도를 높일 수 있습니다."
            )

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """<style>
    .stChatMessage { border-radius: 10px; padding: 10px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stChatMessage[data-testid="stChatMessage-user"] { background-color: #e1f5fe; }
    .stChatMessage[data-testid="stChatMessage-assistant"] { background-color: #f1f8e9; }
</style>""",
    unsafe_allow_html=True,
)

provider = st.session_state.get("model_provider", "OpenAI")
model_name = st.session_state.get("selected_model", AVAILABLE_OPENAI_MODELS[0])

if provider == "Ollama":
    llm = ChatOllama(model=model_name, temperature=0)
else:
    llm = ChatOpenAI(model=model_name, temperature=0, max_retries=3, streaming=True)

graph = create_supervisor_graph(llm)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "안녕하세요! 위에서 대화 모드를 선택하고 질문해주세요.",
        }
    ]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit-thread-{os.urandom(4).hex()}"

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "pdf_download" in msg:
            pdf_info = msg["pdf_download"]
            st.download_button(
                label=f"'{pdf_info['name']}' 다운로드 📥",
                data=pdf_info['bytes'],
                file_name=pdf_info['name'],
                mime="application/pdf",
                use_container_width=True,
                key=f"download_{st.session_state.thread_id}_{i}" # 고유 키 생성
            )

if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        status_container = st.expander("에이전트 작업 과정", expanded=True)
        answer_placeholder = st.empty()

        config = RunnableConfig(
            recursion_limit=RECURSION_LIMIT,
            configurable={
                "thread_id": st.session_state.thread_id,
                "user_id": st.session_state.user_id,
                "is_admin": st.session_state.is_admin,
            },
        )

        conversation_history = [
            (
                HumanMessage(content=msg["content"])
                if msg["role"] == "user"
                else AIMessage(content=msg["content"])
            )
            for msg in st.session_state.messages
        ]
        chat_mode = st.session_state.get("chat_mode")
        inputs = {"messages": conversation_history}
        if "last_sql_result" in st.session_state:
            inputs["result"] = st.session_state.last_sql_result

        stream_kwargs = {"config": config, "version": "v2"}

        if chat_mode != "Supervisor":
            stream_kwargs["configurable"] = {**config.get("configurable", {}), "next": [chat_mode]}

        async def run_and_stream():
            """그래프를 실행하고 UI에 실시간으로 스트리밍합니다."""
            final_answer = ""
            pdf_info_for_message = None  # PDF 정보를 임시 저장할 변수

            log_placeholder = status_container.empty()
            tool_output_placeholder = status_container.empty()

            log_content = ["🚀 워크플로우 시작..."]
            log_placeholder.markdown("\n\n".join(log_content), unsafe_allow_html=True)

            current_agent = ""
            
            async for event in graph.astream_events(inputs, **stream_kwargs):
                kind = event["event"]

                if kind == "on_chain_start":
                    if event["name"] in ["Supervisor", "SQLAgent", "RAGAgent", "ReportAgent", "VisualizationAgent"]:
                        current_agent = event["name"]
                    if event["name"] == "answer_generator":
                        final_answer = ""

                elif kind == "on_tool_start":
                    tool_output_placeholder.empty()
                    log_content.append(f"▶️ **{current_agent}**가 **{event['name']}** 도구를 사용하는 중...")
                    log_placeholder.markdown("\n\n".join(log_content), unsafe_allow_html=True)

                elif kind == "on_tool_end":
                    if log_content and " 중..." in log_content[-1]:
                        log_content.pop()
                    log_content.append(f"✅ **{current_agent}**가 **{event['name']}** 도구를 사용했습니다.")
                    log_placeholder.markdown("\n\n".join(log_content), unsafe_allow_html=True)

                    if event["name"] == "execute_query":
                        output = event["data"]["output"]
                        st.session_state.last_sql_result = output
                        with tool_output_placeholder.container():
                            try:
                                result_data = json.loads(output)
                                if "error" in result_data:
                                    st.error(f"**Query Error ❌**\n```\n{result_data['error']}\n```")
                                elif "data" in result_data:
                                    st.markdown("**Query Result 📝**")
                                    df = pd.DataFrame(result_data.get("data", []), columns=result_data.get("columns", []))
                                    st.dataframe(df, use_container_width=True)
                                    if result_data.get("truncated"):
                                        st.info(f"결과가 너무 많아 최대 {len(df)}건만 표시합니다.")
                            except (json.JSONDecodeError, TypeError):
                                st.markdown(f"**Result**\n```\n{output}\n```")
                    elif event["name"] == "create_pdf_report":
                        output = event["data"]["output"]
                        if "Successfully generated professional PDF report:" in output:
                            filename = output.split(":", 1)[1].strip()
                            if os.path.exists(filename):
                                with open(filename, "rb") as f:
                                    pdf_bytes = f.read()
                                pdf_info_for_message = {"name": filename, "bytes": pdf_bytes}
                                with tool_output_placeholder.container():
                                    st.success(f"'{filename}' 보고서가 성공적으로 생성되었습니다.")
                    elif event["name"] == "create_line_chart":
                        pass

                elif kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        is_supervisor_acting_as_rag = current_agent == "Supervisor" and not chunk.tool_calls
                        if current_agent != "Supervisor" or is_supervisor_acting_as_rag:
                            if final_answer or not chunk.content.isspace():
                                final_answer += chunk.content
                                answer_placeholder.markdown(final_answer + "▌")

            log_content.append("🏁 워크플로우 종료.")
            log_placeholder.markdown("\n\n".join(log_content), unsafe_allow_html=True)
            answer_placeholder.markdown(final_answer)
            
            new_message = {"role": "assistant", "content": final_answer}
            if pdf_info_for_message:
                new_message["pdf_download"] = pdf_info_for_message
            st.session_state.messages.append(new_message)

        try:
            asyncio.run(run_and_stream())
            st.rerun() # Re-run to display the new message and download button
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            error_message = "죄송합니다, 처리 중 오류가 발생했습니다."
            st.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

