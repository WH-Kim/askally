# modules/graph.py

import functools
import re
import json
from typing import Literal
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .state import AgentState
from .agents import (
    direct_response_node,
    rag_agent_node,
    chart_generator_agent_node,
    report_agent_node,
    text_to_sql_query,
)
from .tools import execute_sql_and_get_results, list_tables, get_schema

# --- SQL Agent Sub-Graph ---
def create_sql_agent_graph(llm, db_info):
    
    # --- FIX ---
    # 답변 생성 프롬프트 수정: 데이터가 10건으로 제한되었음을 명시하도록 가이드 추가
    answer_generator_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 데이터 분석가입니다. 사용자의 질문과 제공된 SQL 쿼리 결과를 바탕으로 간결하고 자연스러운 한국어 답변을 작성하세요.\n"
                   "**중요**: 제공된 데이터는 최대 10건으로 제한되어 있습니다. 답변 시 '데이터는 최대 10건만 조회되었습니다.'와 같이 이 점을 반드시 언급해주세요.\n"
                   "원래 질문: {question}"),
        ("human", "쿼리 결과:\n{result}"),
    ])
    answer_generator_chain = answer_generator_prompt | llm

    # --- Sub-Graph Nodes ---
    def get_tables_node(state: AgentState):
        return {"table_names": list_tables.invoke({})}

    def select_tables_node(state: AgentState):
        all_table_names = state["table_names"]

        table_selector_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at selecting relevant SQL tables. "
                       "Based on the user's question, return a comma-separated list of table names that are relevant. "
                       "IMPORTANT: Only choose from the following list of available tables: {table_list}. "
                       "Do not return any table name that is not in this list."),
            MessagesPlaceholder(variable_name="messages"),
        ])
        table_selector = table_selector_prompt | llm

        if not all_table_names: 
            return {"schema": "No tables found in the database."}
            
        response = table_selector.invoke({
            "table_list": ", ".join(all_table_names),
            "messages": state["messages"]
        })
        
        suggested_tables = [t.strip() for t in response.content.split(",")]
        validated_tables = [table for table in suggested_tables if table in all_table_names]
        
        return {"table_names": validated_tables}


    def get_schema_node(state: AgentState):
        if not state["table_names"]: return {"schema": "관련 테이블이 선택되지 않았습니다."}
        return {"schema": get_schema.invoke({"table_names": ",".join(state["table_names"])})}

    def query_gen_node(state: AgentState):
        if not state.get("schema") or "관련 테이블이 선택되지 않았습니다" in state.get("schema"):
            return {"query": "SELECT '관련 테이블을 찾지 못해 쿼리를 생성할 수 없습니다.'"}
            
        query = text_to_sql_query(
            messages=state["messages"],
            llm=llm,
            table_info=state["schema"],
            few_shot_examples=db_info.get("few_shot_examples", [])
        )
        return {"query": query}

    def execute_query_node(state: AgentState):
        try:
            result = execute_sql_and_get_results.invoke({"query": state["query"]})
        except Exception as e:
            result = f'{{"error": "{e}"}}'
        return {"result": result}
    
    def answer_generator_node(state: AgentState):
        last_question = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_question = msg.content
                break
                
        stream = answer_generator_chain.stream({
            "question": last_question,
            "result": state["result"]
        })
        for chunk in stream:
            yield {"messages": [chunk]}

    def after_execute_router(state: AgentState) -> Literal["answer_generator", "get_tables"]:
        try:
            result_data = json.loads(state.get("result", "{}"))
            if "error" in result_data:
                return "get_tables" 
        except (json.JSONDecodeError, TypeError):
            pass
        return "answer_generator"

    workflow = StateGraph(AgentState)
    workflow.add_node("get_tables", get_tables_node)
    workflow.add_node("select_tables", select_tables_node)
    workflow.add_node("get_schema", get_schema_node)
    workflow.add_node("query_gen", query_gen_node)
    workflow.add_node("execute_query", execute_query_node)
    workflow.add_node("answer_generator", answer_generator_node)
    
    workflow.add_edge(START, "get_tables")
    workflow.add_edge("get_tables", "select_tables")
    workflow.add_edge("select_tables", "get_schema")
    workflow.add_edge("get_schema", "query_gen")
    workflow.add_edge("query_gen", "execute_query")
    workflow.add_conditional_edges("execute_query", after_execute_router)
    workflow.add_edge("answer_generator", END)
    
    return workflow.compile()

# --- Main Supervisor Graph ---
def supervisor_node(state: AgentState):
    if not isinstance(state["messages"][-1], HumanMessage):
        return {"next": "FINISH"}
        
    user_query = state["messages"][-1].content.lower()
    
    if any(k in user_query for k in ['보고서', 'pdf', '출력', '저장', '정리']):
        return {"next": "ReportAgent"}
    if any(k in user_query for k in ['차트', '그래프', '시각화', 'plot', 'graph', 'chart']):
        return {"next": "ChartGeneratorAgent"}
    if any(k in user_query for k in ['db', '데이터베이스', '고객', '직원', '테이블', 'sql', '순위']):
        return {"next": "SQLAgent"}
    if any(k in user_query for k in ['문서', '파일', '제도', '기준']):
        return {"next": "RAGAgent"}
    return {"next": "DirectResponse"}

def create_supervisor_graph(llm, retriever, db_info):
    sql_sub_graph = create_sql_agent_graph(llm, db_info)
    rag_node = functools.partial(rag_agent_node, llm=llm, retriever=retriever)
    direct_node = functools.partial(direct_response_node, llm=llm)
    chart_node = functools.partial(chart_generator_agent_node, llm=llm, db_info=db_info)
    report_node = functools.partial(report_agent_node, llm=llm)

    workflow = StateGraph(AgentState)
    workflow.add_node("RAGAgent", rag_node)
    workflow.add_node("DirectResponse", direct_node)
    workflow.add_node("SQLAgent", sql_sub_graph)
    workflow.add_node("ChartGeneratorAgent", chart_node)
    workflow.add_node("ReportAgent", report_node)
    workflow.add_node("Supervisor", supervisor_node)
    
    def entry_router(state: AgentState): 
        return state.get("next") or "Supervisor"
    
    workflow.add_conditional_edges(START, entry_router, {
        "Supervisor": "Supervisor", "RAGAgent": "RAGAgent", "SQLAgent": "SQLAgent",
        "ChartGeneratorAgent": "ChartGeneratorAgent", "ReportAgent": "ReportAgent", "DirectResponse": "DirectResponse",
    })
    workflow.add_conditional_edges("Supervisor", lambda x: x["next"], {
        "SQLAgent": "SQLAgent", "RAGAgent": "RAGAgent", "ChartGeneratorAgent": "ChartGeneratorAgent",
        "ReportAgent": "ReportAgent", "DirectResponse": "DirectResponse", "FINISH": END,
    })
    
    workflow.add_edge("RAGAgent", END)
    workflow.add_edge("SQLAgent", END)
    workflow.add_edge("DirectResponse", END)
    workflow.add_edge("ChartGeneratorAgent", END)
    workflow.add_edge("ReportAgent", END)
    
    return workflow.compile()