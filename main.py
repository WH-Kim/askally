from dataclasses import dataclass
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote import logging
from langchain_teddynote.messages import random_uuid
from dotenv import load_dotenv
from modules.handler import stream_handler, format_search_result
from modules.tools import WebSearchTool
from modules.db_tools import get_db_tools

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름
logging.langsmith("Ask-Ally")

st.title("Ask-Ally 💬")
st.markdown(
    """
    **NH Digital-X**
    #### 무엇이든 물어보올리
    ---
    *오원철, 류호찬, 홍석영, 김원현*
    """
)

# 대화기록을 저장하기 위한 용도로 생성
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ReAct Agent 초기화
if "react_agent" not in st.session_state:
    st.session_state["react_agent"] = None

# include_domains 초기화
if "include_domains" not in st.session_state:
    st.session_state["include_domains"] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    st.markdown("Ask-Ally")

    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)

    # 검색 결과 개수 설정
    search_result_count = st.slider("검색 결과", min_value=1, max_value=10, value=3)

    # include_domains 설정
    st.subheader("검색 도메인 설정")
    search_topic = st.selectbox("검색 주제", ["general", "news"], index=0)
    new_domain = st.text_input("추가할 도메인 입력")
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("도메인 추가", key="add_domain"):
            if new_domain and new_domain not in st.session_state["include_domains"]:
                st.session_state["include_domains"].append(new_domain)

    st.divider()

    st.subheader("DB 연결 설정")
    db_type = st.selectbox("DB 유형", ["SQLite", "MySQL"], key="db_type")

    if st.session_state.db_type == "SQLite":
        db_path = st.text_input("SQLite DB 경로", "sqlite.db", key="db_path")
        st.session_state["db_info"] = {"type": "sqlite", "path": db_path}
    elif st.session_state.db_type == "MySQL":
        mysql_host = st.text_input("MySQL Host", "localhost", key="mysql_host")
        mysql_user = st.text_input("MySQL User", "root", key="mysql_user")
        mysql_password = st.text_input(
            "MySQL Password", type="password", key="mysql_password"
        )
        mysql_db = st.text_input("MySQL DB Name", "sakila", key="mysql_db")
        st.session_state["db_info"] = {
            "type": "mysql",
            "host": mysql_host,
            "user": mysql_user,
            "password": mysql_password,
            "db": mysql_db,
        }
    # 현재 등록된 도메인 목록 표시
    st.write("등록된 도메인 목록:")
    for idx, domain in enumerate(st.session_state["include_domains"]):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(domain)
        with col2:
            if st.button("삭제", key=f"del_{idx}"):
                st.session_state["include_domains"].pop(idx)
                st.rerun()

    # 설정 버튼
    apply_btn = st.button("설정 완료", type="primary")


@dataclass
class ChatMessageWithType:
    chat_message: ChatMessage
    msg_type: str
    tool_name: str


# 이전 대화를 출력
def print_messages():
    for message in st.session_state["messages"]:
        if message.msg_type == "text":
            st.chat_message(message.chat_message.role).write(
                message.chat_message.content
            )
        elif message.msg_type == "tool_result":
            with st.expander(f"✅ {message.tool_name}"):
                st.markdown(message.chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message, msg_type="text", tool_name=""):
    if msg_type == "text":
        st.session_state["messages"].append(
            ChatMessageWithType(
                chat_message=ChatMessage(role=role, content=message),
                msg_type="text",
                tool_name=tool_name,
            )
        )
    elif msg_type == "tool_result":
        content = ""
        if tool_name == "web_search":
            content = format_search_result(message)
        else:
            # For other tools like SQL, just use the raw message
            content = f"```sql\n{message}\n```"

        st.session_state["messages"].append(
            ChatMessageWithType(
                chat_message=ChatMessage(role="assistant", content=content),
                msg_type="tool_result",
                tool_name=tool_name,
            )
        )


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["thread_id"] = random_uuid()
# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 설정 버튼이 눌리면...
if apply_btn:
    tools = []
    # WebSearchTool 생성
    web_tool = WebSearchTool(
        max_results=search_result_count,
        include_domains=st.session_state["include_domains"],
        topic=search_topic,
    ).create()
    tools.append(web_tool)

    # DB Tool 생성
    if "db_info" in st.session_state:
        try:
            db_tools = get_db_tools(
                st.session_state["db_info"], model_name=selected_model
            )
            tools.extend(db_tools)
            st.success("DB 연결에 성공했습니다.")
        except Exception as e:
            st.error(f"DB 연결에 실패했습니다: {e}")
    st.session_state["react_agent"] = create_agent_executor(
        model_name=selected_model, tools=tools
    )
    st.session_state["thread_id"] = random_uuid()

# 만약에 사용자 입력이 들어오면...
if user_input:
    agent = st.session_state["react_agent"]
    # Config 설정

    if agent is not None:
        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        # 사용자의 입력
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            (
                container_messages,
                tool_args,
                agent_answer,
            ) = stream_handler(  # stream_handler 인수를 올바르게 수정
                container,
                agent,
                {
                    "messages": [
                        ("human", user_input),
                    ]
                },
                config,
            )

            # 대화기록을 저장한다.
            add_message("user", user_input)
            for tool_arg in tool_args:
                add_message(
                    "assistant",
                    tool_arg["tool_result"],
                    "tool_result",
                    tool_arg["tool_name"],
                )
            add_message("assistant", agent_answer)
    else:
        warning_msg.warning("사이드바에서 설정을 완료해주세요.")
