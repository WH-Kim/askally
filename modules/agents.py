# modules/agents.py

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.prebuilt import create_react_agent
from .state import AgentState
from .tools import create_rag_retriever, get_sql_tools
import streamlit as st

def pdf_rag_node(state: AgentState, llm, name: str = "PDF_RAG_Agent"):
    messages = state["messages"]
    # 시스템 메시지를 제외한 마지막 사용자 메시지를 질문으로 사용
    question = [msg.content for msg in messages if isinstance(msg, HumanMessage)][-1]
    
    # st.session_state.vector_store는 app.py에서 로드됩니다.
    retriever = create_rag_retriever(st.session_state.vector_store)
    if not retriever:
        return {"messages": [HumanMessage(content="RAG 벡터 데이터베이스가 준비되지 않았습니다. `rag_documents` 폴더를 확인해주세요.", name=name)]}
        
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template("다음 컨텍스트를 사용하여 질문에 답하세요:\n\n{context}\n\n질문: {question}")
        | llm
    )
    
    response = rag_chain.invoke(question)
    return {"messages": [HumanMessage(content=response.content, name=name)]}

def direct_response_node(state: AgentState, llm, name: str = "DirectResponse"):
    # 이전 대화 기록을 포함하여 전달
    response = llm.invoke(state["messages"])
    return {"messages": [HumanMessage(content=response.content, name=name)]}

@st.cache_resource
def create_sql_agent_runnable(_llm):
    sql_tools = get_sql_tools(_llm)
    return create_react_agent(_llm, tools=sql_tools)

def sql_agent_node(state: AgentState, llm, name="SQLAgent"):
    few_shot_examples = st.session_state.get("few_shot_examples", [])
    example_str = "\n\n---\n\n".join([f"User Question: {ex['question']}\nSQL Query: {ex['query']}" for ex in few_shot_examples])
    
    system_prompt = (
        "You are an expert SQL agent. Your primary goal is to answer the user's question by interacting with a database.\n\n"
        "## Your Thought Process (ReAct Framework):\n"
        "1.  **Thought:** Analyze the user's question. Decide if you need to explore the database schema first or if you can write a query directly.\n"
        "2.  **Action:** Use `sql_db_list_tables` or `sql_db_schema` for exploration. Use `sql_db_query` to get the answer.\n"
        "3.  **Observation:** Look at the result from the tool.\n"
        "4.  **Thought:** Based on the observation, decide your next step. Repeat until you have the final answer.\n"
        "5.  Once you have the final data, formulate a clear, natural language response to the user.\n\n"
        f"## Few-Shot Examples (Use these to understand the data):\n{example_str}"
    )
    
    # ReAct 에이전트가 이전 대화 기록도 참고할 수 있도록 전체 메시지를 전달합니다.
    new_messages = [HumanMessage(content=system_prompt)] + state['messages']
    
    new_state = state.copy()
    new_state['messages'] = new_messages
    
    agent_runnable = create_sql_agent_runnable(llm)
    result = agent_runnable.invoke(new_state)
    
    # ReAct 에이전트의 전체 메시지(생각 과정 포함)를 반환하여 스트리밍 시각화에 사용
    return {"messages": result["messages"]}