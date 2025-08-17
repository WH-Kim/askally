# modules/graph.py

import functools
import re
import json
from typing import Literal, List
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .state import AgentState
from .agents import create_rag_agent_executor, agent_node, direct_response_node
from .tools import execute_query, list_tables, get_schema

def create_sql_agent_graph(llm):
    table_selector = (ChatPromptTemplate.from_messages([
        ("system", "Return a comma-separated list of tables relevant to the user's question from: {table_list}"),
        MessagesPlaceholder(variable_name="messages"),
    ]) | llm)

    query_gen_chain = (ChatPromptTemplate.from_messages([
        ("system", """You are a SQL expert. Based on the user's question and the database schema, generate a correct SQLite query.
If a previous attempt resulted in an error, use the error message to fix the query.
**IMPORTANT**: Your output must be ONLY the raw SQL query, without any additional explanation or markdown formatting."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Schema: {schema}\n\nPrevious Result or Error:\n{result}"),
    ]) | llm)
    
    answer_generator = (ChatPromptTemplate.from_messages([
        ("system", "You are a data analyst. Based on the user's question and the provided SQL query result, formulate a concise and natural answer in Korean.\nOriginal Question: {question}"),
        ("human", "Query Result:\n{result}"),
    ]) | llm)

    def get_tables_node(state: AgentState):
        tables = list_tables.invoke({})
        return {"table_names": tables}

    def select_tables_node(state: AgentState):
        if not state["table_names"]: return {"schema": "No tables found in DB."}
        response = table_selector.invoke({"table_list": ", ".join(state["table_names"]), "messages": state["messages"][-1:]})
        return {"table_names": [t.strip() for t in response.content.split(",")]}

    def get_schema_node(state: AgentState):
        if not state["table_names"]: return {"schema": "No relevant tables selected."}
        schema = get_schema.invoke({"table_names": ",".join(state["table_names"])})
        return {"schema": schema}

    def query_gen_node(state: AgentState):
        response = query_gen_chain.invoke({"schema": state["schema"], "messages": state["messages"], "result": state.get("result", "")})
        return {"messages": state["messages"] + [response]}
    
    def query_parser_node(state: AgentState):
        last_message = state["messages"][-1].content
        query = last_message.strip()
        match = re.search(r"```sql\n(.*?)\n```", query, re.DOTALL)
        if match: query = match.group(1).strip()
        return {"query": query}

    def execute_query_node(state: AgentState):
        try:
            result = execute_query.invoke({"query": state["query"]})
        except Exception as e:
            result = f'{{"error": "{e}"}}'
        return {"result": result}

    def answer_generator_node(state: AgentState):
        original_question = state["messages"][0].content
        response = answer_generator.invoke({"question": original_question, "result": state["result"]})
        return {"messages": [AIMessage(content=response.content, name="SQLAgent")]}

    def after_execute_router(state: AgentState) -> Literal["answer_generator", "query_gen"]:
        try:
            result_data = json.loads(state.get("result", "{}"))
            if "error" in result_data: return "query_gen"
        except (json.JSONDecodeError, TypeError):
            if "Error:" in str(state.get("result", "")): return "query_gen"
        return "answer_generator"

    workflow = StateGraph(AgentState)
    workflow.add_node("get_tables", get_tables_node)
    workflow.add_node("select_tables", select_tables_node)
    workflow.add_node("get_schema", get_schema_node)
    workflow.add_node("query_gen", query_gen_node)
    workflow.add_node("query_parser", query_parser_node)
    workflow.add_node("execute_query", execute_query_node)
    workflow.add_node("answer_generator", answer_generator_node)

    workflow.add_edge(START, "get_tables")
    workflow.add_edge("get_tables", "select_tables")
    workflow.add_edge("select_tables", "get_schema")
    workflow.add_edge("get_schema", "query_gen")
    workflow.add_edge("query_gen", "query_parser")
    workflow.add_edge("query_parser", "execute_query")
    workflow.add_conditional_edges("execute_query", after_execute_router)
    workflow.add_edge("answer_generator", END)
    return workflow.compile()

def supervisor_node(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.name in ["SQLAgent", "RAGAgent", "DirectResponse"]: return {"next": "FINISH"}
    if isinstance(last_message, HumanMessage):
        user_query = last_message.content
        if any(keyword in user_query.lower() for keyword in ['db', '데이터베이스', '고객', '직원', '테이블', 'sql']):
            return {"next": "SQLAgent"}
        elif any(keyword in user_query.lower() for keyword in ['pdf', '문서', '파일', '고향사랑']):
            return {"next": "RAGAgent"}
        else:
            return {"next": "DirectResponse"}
    return {"next": "FINISH"}

def create_supervisor_graph(llm):
    rag_agent_executor = create_rag_agent_executor(llm)
    rag_agent = functools.partial(agent_node, agent=rag_agent_executor, name="RAGAgent")
    direct_node = functools.partial(direct_response_node, llm=llm)
    sql_sub_graph = create_sql_agent_graph(llm)

    workflow = StateGraph(AgentState)
    workflow.add_node("RAGAgent", rag_agent)
    workflow.add_node("DirectResponse", direct_node)
    workflow.add_node("SQLAgent", sql_sub_graph)
    workflow.add_node("Supervisor", supervisor_node)

    def entry_router(state: AgentState): return state.get("next") or "Supervisor"
    
    workflow.add_conditional_edges(
        "__start__",
        entry_router,
        {
            "Supervisor": "Supervisor",
            "RAGAgent": "RAGAgent",
            "SQLAgent": "SQLAgent",
            "DirectResponse": "DirectResponse",
        }
    )
    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next"],
        {
            "SQLAgent": "SQLAgent",
            "RAGAgent": "RAGAgent",
            "DirectResponse": "DirectResponse",
            "FINISH": END,
        }
    )

    workflow.add_edge("RAGAgent", END)
    workflow.add_edge("SQLAgent", END)
    workflow.add_edge("DirectResponse", END)
    return workflow.compile()