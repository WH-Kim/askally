# modules/graph.py

import functools
import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from .state import AgentState
from .agents import sql_agent_node, pdf_rag_node, direct_response_node

class Route(BaseModel):
    next: str = Field(enum=["SQLAgent", "PDF_RAG_Agent", "DirectResponse", "FINISH"])

@st.cache_resource
def create_supervisor_graph(_llm):
    # SQLAgent의 최종 출력을 Supervisor가 이해할 수 있도록 래핑하는 함수
    def wrapped_sql_node(state, llm):
        result = sql_agent_node(state, llm)
        # ReAct의 최종 답변(AIMessage)을 HumanMessage로 변환하여 Supervisor에게 전달
        final_answer = result["messages"][-1].content
        return {"messages": [HumanMessage(content=final_answer, name="SQLAgent")]}

    sql_node_with_llm = functools.partial(wrapped_sql_node, llm=_llm)
    pdf_node_with_llm = functools.partial(pdf_rag_node, llm=_llm)
    direct_node_with_llm = functools.partial(direct_response_node, llm=_llm)
    
    members = ["SQLAgent", "PDF_RAG_Agent"]
    system_prompt = (
        "당신은 다음과 같은 전문가 에이전트 팀을 관리하는 감독자입니다: {members}. "
        "사용자의 요청의 의도를 명확히 파악하여 가장 적합한 작업자를 선택하세요.\n\n"
        "## 라우팅 규칙:\n"
        "- 사용자가 'PDF', '문서', '파일', '읽어줘' 같은 단어를 사용하면 **'PDF_RAG_Agent'**를 선택하세요.\n"
        "- 사용자가 '데이터베이스', 'DB', '테이블', '고객', '직원', '조회', '검색', '쿼리' 같은 단어를 사용하면 **'SQLAgent'**를 선택하세요.\n"
        "- 위 규칙에 해당하지 않는 간단한 인사, 일반적인 대화, 창의적인 질문 등은 **'DirectResponse'**를 선택하여 직접 답변하세요.\n"
        "- 전문가의 답변이 끝나면 사용자가 만족했는지 확인하고, 더 이상 질문이 없으면 **'FINISH'**를 선택하여 대화를 종료하세요."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "주어진 대화 내용을 바탕으로, 다음에 행동할 작업자를 선택하세요: {options}"),
    ]).partial(options=str(members + ["DirectResponse", "FINISH"]), members=", ".join(members))

    supervisor_chain = prompt | _llm.with_structured_output(Route)

    workflow = StateGraph(AgentState)
    workflow.add_node("Supervisor", supervisor_chain)
    workflow.add_node("SQLAgent", sql_node_with_llm)
    workflow.add_node("PDF_RAG_Agent", pdf_node_with_llm)
    workflow.add_node("DirectResponse", direct_node_with_llm)

    workflow.add_edge(START, "Supervisor")
    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next"],
        {"SQLAgent": "SQLAgent", "PDF_RAG_Agent": "PDF_RAG_Agent", "DirectResponse": "DirectResponse", "FINISH": END},
    )
    workflow.add_edge("SQLAgent", "Supervisor")
    workflow.add_edge("PDF_RAG_Agent", "Supervisor")
    workflow.add_edge("DirectResponse", "Supervisor")

    return workflow.compile()