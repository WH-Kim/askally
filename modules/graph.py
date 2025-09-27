# modules/graph.py

import functools
import re
import json
import pandas as pd
from typing import Literal, List
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .state import AgentState
from .agents import (
    create_rag_agent_executor,
    agent_node,
    create_report_agent_executor,
    create_agent)
from .tools import execute_query, list_tables, get_schema, create_line_chart
from langchain_core.runnables import RunnablePassthrough
from .agents import create_rag_agent_executor
from .utils import load_db_schema_descriptions


def create_sql_agent_graph(llm):
    table_selector_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at analyzing database schemas. Your task is to identify which tables are relevant to answer a user's question.\nBased on the database schema provided below and the user's question, return a comma-separated list of the most relevant table names.\n\nThe schema may include Korean descriptions for tables and columns under the '테이블 및 컬럼 한국어 설명' section.\nYou MUST prioritize these Korean descriptions to understand the user's intent in Korean.\n\n**Database Schema:**\n{db_schema}\n\n**Important Rules:**\n- Only return table names that are present in the provided schema.\n- If multiple tables are relevant (e.g., for a JOIN operation), include all of them.\n- If no tables seem relevant, return an empty string.\n- Your output MUST be a single line of comma-separated table names. For example: `table1, table2`\n""",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    table_selector = table_selector_prompt | llm

    query_gen_chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert SQLite data analyst. Your task is to generate a single, correct SQLite query based on the user's question and the provided database schema.\n\nThe schema may include Korean descriptions for tables and columns under the '테이블 및 컬럼 한국어 설명' section.\nYou MUST use these Korean descriptions to correctly map the user's Korean question to the appropriate tables and columns.\n\n**Query Generation Guidelines:**\n1.  **JOINs and Aliases **: When a query requires joining tables, you MUST use clear aliases for each table (e.g., `t1`, `t2`). All column names in `SELECT`, `WHERE`, `GROUP BY`, and `ORDER BY` clauses MUST be prefixed with their corresponding table alias (e.g., `t1.column_name`) to prevent \"ambiguous column\" errors.\n2.  **Correct JOIN Conditions**: Carefully examine the schema to determine the correct columns for joining tables.\n3.  **String Matching**: For text searches, use the `LIKE` operator with wildcards (`%`). For example, to find a name containing \"영업\", use `WHERE name LIKE '%영업%'`.\n4.  **Aggregations**: When using aggregate functions (`AVG`, `SUM`, `COUNT`, etc.) with other columns, ensure you use a `GROUP BY` clause for all non-aggregated columns.\n5.  **SQLite Syntax**: Remember that this is for SQLite. Pay attention to its specific syntax, especially for date and time functions (e.g., `strftime`).\n\n**General Performance Queries:**\n- If the user asks for their performance data (e.g., \"내 실적 보여줘\", \"내 점수 알려줘\") and the question does NOT contain \"보고서\" or \"리포트\", you should retrieve the raw data as is.\n- Do NOT calculate changes from the previous period (using `LAG` or any other method) unless the user specifically asks for a \"보고서\" (report).\n\n**Chart-Specific Queries:**\n- If the user asks for a trend chart of their score (e.g., '총점', 'EMP_TOT_SCR'), you MUST generate a query that provides not only the user's score but also the averages for their business division, job grade, and the national average.\n- Use the following query structure as a template. You MUST replace '{user_id}' with the actual user ID.\n\n```sql\nWITH UserData AS (\n    SELECT NAME, BAS_DT, PROV_C, PZCNM, EMP_TOT_SCR\n    FROM TB_BESTBANKER\n    WHERE ENO = '{user_id}'\n)\nSELECT\n    ud.BAS_DT,\n    ud.EMP_TOT_SCR AS \"내 총점\",\n    (SELECT AVG(EMP_TOT_SCR) FROM TB_BESTBANKER WHERE BAS_DT = ud.BAS_DT) AS \"전국 평균\",\n    (SELECT AVG(EMP_TOT_SCR) FROM TB_BESTBANKER WHERE BAS_DT = ud.BAS_DT AND PROV_C = ud.PROV_C) AS \"영업본부 평균\",\n    (SELECT AVG(EMP_TOT_SCR) FROM TB_BESTBANKER WHERE BAS_DT = ud.BAS_DT AND PZCNM = ud.PZCNM) AS \"직급 평균\"\nFROM UserData ud\nORDER BY ud.BAS_DT\n```\n\n**Report-Specific Queries:**\n- If the user's latest question contains \"보고서\" or \"리포트\" (report), you MUST generate a comprehensive query to retrieve all data necessary for the 'Best Banker Performance Analysis Report'.\n- The query MUST be for the specified user ID ('{user_id}').\n- It MUST retrieve the most recent record for that user by ordering by `BAS_DT` descending and taking the top 1.\n- It MUST retrieve all 12 performance category scores (e.g., `PSN_LN_SCR`, `CORP_LN_SCR`, etc.) and their corresponding ranks (e.g., `PSN_LN_ORD`, `CORP_LN_ORD`, etc.).\n- Crucially, it MUST also calculate the change in score from the previous period for each of the 12 categories. Use the `LAG` window function to get the previous score. For example, `(t1.PSN_LN_SCR - LAG(t1.PSN_LN_SCR, 1, 0) OVER (PARTITION BY t1.ENO ORDER BY t1.BAS_DT))` as `개인여신증감`. You must do this for all 12 score columns.\n- The final SELECT statement should retrieve the employee's name, branch name, position, the evaluation date, and all 12 scores, ranks, and score changes.\n\nIf a previous attempt resulted in an error, use the error message to fix the query.\n\n**IMPORTANT**: Your output must be ONLY the raw SQL query, without any additional explanation or markdown formatting (e.g., no ```sql ... ```).\n\n**USER CONTEXT**:\n- User ID: {user_id}\n- User Role: {user_role}\n\nIf the user's role is 'user', all queries on the `TB_BESTBANKER` table MUST include a `WHERE eno = '{user_id}'` clause.\nIf the user's role is 'admin', this filter is not necessary.""",
                ),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "Schema: {schema}\n\nPrevious Result or Error:\n{result}"),
            ]
        )
        | llm
    )

    # DB 스키마 설명을 불러와 프롬프트에 포함
    try:
        schema_descriptions = json.dumps(load_db_schema_descriptions(), ensure_ascii=False, indent=2)
    except Exception:
        schema_descriptions = "{}"

    answer_generator_prompt = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful data analyst. Your goal is to provide a clear and concise answer in Korean based on the user's question and the SQL query results.
You MUST use the provided Korean schema descriptions to refer to tables and columns in your answer. For example, instead of saying "CD_SCR", you should say "카드 점수".

    **Korean Schema Descriptions:**
    ```json
    {schema_descriptions}
    ```

    **Analysis Guidelines:**
    - Analyze the provided SQL query result in the context of the original question.
    - If the result contains data, summarize it clearly and answer the user's question using the Korean column names from the schema descriptions.
    - If the result is empty, state that no data was found that matches the criteria.
    - Do not mention the SQL query itself in your final answer. Just provide the answer based on the data.
    - **IMPORTANT**: When analyzing data related to rankings (e.g., columns ending with `_ORD` like `NATL_ORD`, `PROV_ORD`), remember that a **lower number indicates a better rank**. For example, a change from rank 10 to 5 is an improvement. Frame your analysis accordingly (e.g., "순위가 상승했습니다" not "순위가 하락했습니다").
    - **CRITICAL**: Do NOT repeat the user's original question in your answer. Only provide the answer based on the data.

    Original Question: {{question}} """,
                ),
                ("human", "Query Result:\n{result}"),
            ]
        )
        | llm
    )    

    def get_schema_node(state: AgentState, config: RunnableConfig):
        """Gets all table names and their schemas to begin the process."""
        user_id = config["configurable"].get("user_id")
        is_admin = config["configurable"].get("is_admin", False)

        all_tables = list_tables.invoke({})
        if not all_tables:
            return {"table_names": [], "schema": "No tables found in the database."}

        schema = get_schema.invoke(
            {"table_names": ",".join(all_tables), "user_id": user_id, "is_admin": is_admin}
        )
        return {"table_names": all_tables, "schema": schema}

    def select_tables_node(state: AgentState, config: RunnableConfig):
        """Selects relevant tables and refines the schema for the query generator."""
        if not state["table_names"]:
            return {
                "table_names": [],
                "schema": "No relevant tables found for the question.",
            }

        user_id = config["configurable"].get("user_id")
        is_admin = config["configurable"].get("is_admin", False)

        response = table_selector.invoke(
            {
                "db_schema": state["schema"],
                "messages": state["messages"][-1:],
            }
        )

        selected_tables_str = response.content.strip()
        if not selected_tables_str:
            return {
                "table_names": [],
                "schema": "No relevant tables found for the question.",
            }

        selected_tables = [t.strip() for t in selected_tables_str.split(",")]
        valid_selected_tables = [
            t for t in selected_tables if t in state["table_names"]
        ]
        refined_schema = get_schema.invoke(
            {
                "table_names": ",".join(valid_selected_tables),
                "user_id": user_id,
                "is_admin": is_admin,
            }
        )
        return {"table_names": valid_selected_tables, "schema": refined_schema}

    def query_gen_node(state: AgentState, config: RunnableConfig):
        user_id = config["configurable"].get("user_id")
        is_admin = config["configurable"].get("is_admin", False)
        user_role = "admin" if is_admin else "user"

        response = query_gen_chain.invoke(
            {
                "schema": state["schema"],
                "messages": state["messages"],
                "result": state.get("result", ""),
                "question": state["messages"][-1].content,
                "user_id": user_id,
                "user_role": user_role,
            }
        )
        return {"messages": state["messages"] + [response]}

    def query_parser_node(state: AgentState):
        last_message = state["messages"][-1].content
        query = last_message.strip()
        match = re.search(r"```sql\n(.*?)\n```", query, re.DOTALL)
        if match:
            query = match.group(1).strip()
        
        # 쿼리 생성 메시지를 messages 리스트에서 제거합니다.
        # 이렇게 하면 answer_generator가 이전 쿼리를 볼 수 없게 됩니다.
        return {
            "query": query,
            "messages": state["messages"][:-1]
        }
    def execute_query_node(state: AgentState, config: RunnableConfig):
        user_id = config["configurable"].get("user_id")
        is_admin = config["configurable"].get("is_admin", False)
        try:
            result = execute_query.invoke({
                "query": state["query"],
                "user_id": user_id,
                "is_admin": is_admin
            })
        except Exception as e:
            result = f'{{"error": "{e}"}}'
        return {"result": result}

    def answer_generator_node(state: AgentState):
        original_question = state["messages"][0].content
        
        # answer_generator는 스트리밍을 지원하지 않으므로 invoke 사용
        response = answer_generator_prompt.invoke(
            {
                "question": original_question,
                "result": state.get("result", "No result found."), # state['result']는 JSON 문자열일 수 있습니다.
                "schema_descriptions": schema_descriptions,
            }
        )
        # SQLAgent의 최종 답변만 messages에 남깁니다.
        # 최종 답변 후에는 불필요한 정보(테이블, 쿼리)가 다시 표시되지 않도록 관련 state를 초기화합니다.
        return {
            "messages": [AIMessage(content=response.content, name="SQLAgent")],
            "result": state["result"],  # 다음 에이전트를 위해 result는 유지
            "query": "",
            "table_names": [],
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("get_schema", get_schema_node)
    workflow.add_node("select_tables", select_tables_node)
    workflow.add_node("query_gen", query_gen_node)
    workflow.add_node("query_parser", query_parser_node)
    workflow.add_node("execute_query", execute_query_node)

    workflow.add_node("answer_generator", answer_generator_node)


    workflow.add_edge("get_schema", "select_tables")
    workflow.add_edge("select_tables", "query_gen")
    workflow.add_edge("query_gen", "query_parser")
    workflow.add_edge("query_parser", "execute_query")
    
    def after_execute_router(state: AgentState) -> Literal["query_gen", "answer_generator"]:
        """쿼리 실행 오류를 확인하는 간단한 라우터."""
        try:
            result_data = json.loads(state.get("result", "{{}}"))
            if "error" in result_data and result_data["error"]:
                return "query_gen"
        except (json.JSONDecodeError, TypeError):
            return "query_gen"
        return "answer_generator"
    workflow.add_conditional_edges("execute_query", after_execute_router, {"query_gen": "query_gen", "answer_generator": "answer_generator"})
    # SQLAgent의 최종 결과를 저장하고 워크플로우를 종료합니다
    workflow.add_edge("answer_generator", END)

    # 시작점을 get_schema로 설정
    workflow.set_entry_point("get_schema")

    return workflow.compile()


def create_visualization_agent_graph(llm):
    """시각화 생성을 위한 에이전트 그래프를 생성합니다."""

    # 1. 차트 생성을 위한 프롬프트
    chart_gen_system_prompt = """You are a data visualization expert. Your task is to generate Python code to create a chart using the `create_line_chart` tool based on the user's request and the provided data.\n\n**Instructions:**\n1.  Analyze the user's question and the provided data to determine the most appropriate chart.\n2.  Identify the columns to be used for the x-axis and y-axis.\n3.  Generate a title for the chart.\n4.  Call the `create_line_chart` tool with the correct parameters (`data_json`, `title`, `x_axis`, `y_axes`).\n    - `data_json` should be the JSON data provided to you.\n    - `title` should be a descriptive title in Korean based on the user's question.\n    - `x_axis` should be the date or time column, typically `BAS_DT`.\n    - `y_axes` should be a list of columns to plot, such as `['NATL_ORD', 'PZC_ORD', 'PROV_ORD']`.\n\n**User's Question:** {question}\n**Data (in JSON format):**\n```json\n{data_json}\n```"""
    chart_gen_prompt = ChatPromptTemplate.from_messages([
        ("system", chart_gen_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    chart_gen_agent = create_agent(llm, [create_line_chart], chart_gen_prompt, is_prompt_object=True)

    # 2. 차트 생성을 위한 노드
    def visualization_node(state: AgentState, config: RunnableConfig):
        """Generates a line chart from the data provided by SQLAgent."""
        question = state["messages"][0].content
        # SQLAgent로부터 전달받은 결과(state['result'])를 사용
        data_json = state.get("result")

        if not data_json:
            return {
                "messages": [AIMessage(content="차트를 생성할 데이터가 없습니다.", name="VisualizationAgent")]
            }

        # 차트 생성 에이전트 호출
        response = chart_gen_agent.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "question": question, "data_json": data_json
            }
        )

        return {"messages": [AIMessage(content=response["output"], name="VisualizationAgent")]}

    workflow = StateGraph(AgentState)
    workflow.add_node("visualize", visualization_node)
    workflow.set_entry_point("visualize")
    workflow.add_edge("visualize", END)
    return workflow.compile()


def create_report_agent_graph(llm):
    """보고서 생성을 위한 에이전트 그래프를 생성합니다."""
    report_agent_executor = create_report_agent_executor(llm)
    report_node = functools.partial(agent_node, agent=report_agent_executor, name="ReportAgent")

    workflow = StateGraph(AgentState)
    workflow.add_node("report", report_node)
    workflow.set_entry_point("report")
    workflow.add_edge("report", END)
    return workflow.compile()


def create_supervisor_graph(llm):
    def _rule_based_router(question: str) -> list[str]:
        question = question.lower()

        # Keywords for complex workflows
        if "차트" in question or "그래프" in question or "시각화" in question:
            return ["SQLAgent", "VisualizationAgent"]
        if "코칭" in question or "컨설팅" in question or "강점" in question or "약점" in question or "개선점" in question:
            return ["SQLAgent", "RAGCoachingAgent"]
        if ("보고서" in question or "종합 분석" in question) and "실적" in question:
            return ["SQLAgent", "ReportAgent"]

        # User-defined keywords
        sql_keywords = ["db", "데이터", "실적", "쿼리"]
        rag_keywords = ["문서", "평가항목", "제도"]
        report_keywords = ["보고서", "종합 분석"]

        if any(keyword in question for keyword in sql_keywords):
            return ["SQLAgent"]
        if any(keyword in question for keyword in rag_keywords):
            return ["RAGSearchAgent"]
        if any(keyword in question for keyword in report_keywords):
            return ["ReportAgent"]
        
        # Fallback to LLM
        supervisor_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at routing a user's request to the correct specialist and deciding the workflow. Your primary language is Korean.
Your goal is to analyze the user's question and route it to the most appropriate agent or sequence of agents.
You have access to the following specialists: 

1.  **SQLAgent**: A database query expert. Use this for any questions related to data stored in the database, such as customer information, employee data, or performance records.
    Example questions: "Show me the employee information for the head office sales department", "What is the average performance score for the marketing department?", "Who are the 5 most recently joined customers?" 
 
2.  **RAGSearchAgent**: An expert at retrieving information from uploaded PDF documents. Use this for questions related to "documents," "evaluation items," "systems," etc.
    Example questions: "Summarize the document about the Hometown Love Donation System", "Find information about the future of AI agents in the PDF file" 
 
3.  **ReportAgent**: An expert at summarizing conversations and generating PDF reports.
    Example questions: "Summarize the conversation so far", "Create a report from the conversation" 
 
4.  **RAGCoachingAgent**: An expert who provides in-depth analysis and coaching based on performance data retrieved by SQLAgent. This agent is not used alone and always runs after SQLAgent.

**Workflow Rules:**
- If the user asks for a chart, graph, or any kind of visualization (e.g., "차트", "그래프", "시각화"), you MUST first route to `SQLAgent` to get the data, and then to `VisualizationAgent` to create the chart. The output for this case should be `SQLAgent,VisualizationAgent`.
- If the user explicitly asks for "coaching," "consulting," "strengths," "weaknesses," or "areas for improvement" regarding their performance, you MUST first route to `SQLAgent` to get the data, and then to `RAGCoachingAgent` for in-depth analysis and coaching. The output for this case should be `SQLAgent,RAGCoachingAgent`.
- For all other cases, choose a single, most appropriate agent. If no specialist is suitable for a general conversation or greeting, return an empty string.

**Output Format:**
- Your output must be a single line containing the name of the chosen agent or a comma-separated list of agents for a workflow.
- Examples: `SQLAgent`, `RAGSearchAgent`, `SQLAgent,VisualizationAgent`, `SQLAgent,RAGCoachingAgent`
Do not add any other text or explanation. If no agent is suitable, return an empty string.""",
                ),
                ("human", "User's question:\n{question}"),
            ]
        )
        
        llm_router = supervisor_prompt | llm
        response = llm_router.invoke({"question": question})
        return [agent.strip() for agent in response.content.strip().split(",") if agent.strip()]

    supervisor_chain = (
        RunnablePassthrough.assign(question=lambda x: x["messages"][-1].content) |
        (lambda data: {"next": _rule_based_router(data["question"]), **data})
    )

    # RAG node for coaching
    def rag_coaching_node(state: AgentState):
        """Performs RAG coaching based on the results from SQLAgent."""
        rag_agent_executor = create_rag_agent_executor(llm)

        sql_result = state.get("result")
        if not sql_result:
            return {"messages": [AIMessage(content="No performance data available for coaching.", name="RAGCoachingAgent")]}

        try:
            schema_descriptions = json.dumps(load_db_schema_descriptions(), ensure_ascii=False, indent=2)
        except Exception:
            schema_descriptions = "{}"

        coaching_prompt = f"""You are a top-tier performance coaching expert for bank employees. Your mission is to provide a personalized and actionable performance improvement plan based on the user's performance data.
When explaining the data, you MUST use the provided Korean schema descriptions to refer to columns. For example, use "카드 점수" instead of "CD_SCR". Remember that for rank columns (`_ORD`), a lower number is a better rank.

**Your Task:**
1.  **Overall Performance Review:**
    -   First, briefly summarize the user's overall performance based on their `PZC_ORD` (직급내순위) and `PROV_ORD` (본부내순위) from the provided data.

2.  **Detailed Analysis of Strengths and Weaknesses:**
    -   Carefully analyze each performance metric in the user's data.
    -   **Strengths**: Identify metrics where the user's `PZC_ORD` and `PROV_ORD` are both in the top tier (e.g., top 30%). These are clear strengths.
    -   **Weaknesses**: Identify metrics where the user's `PZC_ORD` and `PROV_ORD` are both in the bottom tier (e.g., bottom 30%). These are urgent areas for improvement.
    -   **Opportunities**: Note any metrics where there is a significant gap between `PZC_ORD` and `PROV_ORD`, as these might indicate specific competitive landscapes or opportunities.

3.  **Provide Actionable Improvement Strategies for Weaknesses:**
    -   For each identified **weakness**, use the `pdf_document_retriever` tool to find specific, actionable strategies.
    -   Your search query for the tool should be precise. For example, if the weakness is "카드 점수", your query should be "카드 점수 평가 기준" or "카드 점수 배점 방식".
    -   Based on the retrieved information, provide a concrete, step-by-step action plan. For example, instead of saying "improve your card score," explain *how* by referencing the retrieved evaluation criteria, scoring system, and specific actions the user can take (e.g., "To improve your '카드 점수', focus on increasing the '신규 카드 발급 건수' and '카드론 실적'. According to the document, each new card issued contributes 5 points...").

4.  **Synthesize and Structure the Final Report:**
    -   Combine the data analysis and retrieved strategies into a comprehensive coaching report.
    -   Structure your final answer in the following format, using Markdown:
        -   **종합 평가**: A brief summary of the user's current standing.
        -   **강점 분석**: List the identified strengths and briefly explain why they are strengths.
        -   **약점 분석 및 개선 방안**: For each weakness, first state the metric and the user's current rank. Then, provide the detailed, actionable improvement plan derived from the retrieved documents.

**Korean Schema Descriptions:**
```json
{schema_descriptions}
```

**User Performance Data:**
```json
{sql_result}
```"""
        response = rag_agent_executor.invoke({"messages": [HumanMessage(content=coaching_prompt)], **state})
        return {"messages": [AIMessage(content=response["output"], name="RAGCoachingAgent")], "sender": "RAGCoachingAgent"}

    # RAG node for direct document search
    rag_search_agent_executor = create_rag_agent_executor(llm)
    rag_search_node = functools.partial(
        agent_node, agent=rag_search_agent_executor, name="RAGSearchAgent"
    )

    report_graph = create_report_agent_graph(llm)
    sql_sub_graph = create_sql_agent_graph(llm)
    visualization_agent = create_visualization_agent_graph(llm)

    def prepare_report_data_node(state: AgentState):
        """Converts SQL query result from JSON to a markdown table for the report agent."""
        performance_table = "실적 데이터가 없습니다."
        if state.get("result"):
            try:
                result_data = json.loads(state["result"])
                if result_data.get("data"):
                    df = pd.DataFrame(
                        result_data.get("data", []),
                        columns=result_data.get("columns", []),
                    )
                    if not df.empty:
                        performance_table = df.to_markdown(index=False)
            except (json.JSONDecodeError, TypeError):
                performance_table = "실적 데이터 형식에 오류가 있어 표로 변환할 수 없습니다."
        return {"performance_table": performance_table}

    workflow = StateGraph(AgentState)
    workflow.add_node("ReportAgent", report_graph)
    workflow.add_node("RAGSearchAgent", rag_search_node)
    workflow.add_node("RAGCoachingAgent", rag_coaching_node)
    workflow.add_node("SQLAgent", sql_sub_graph)
    workflow.add_node("VisualizationAgent", visualization_agent)
    workflow.add_node("prepare_report_data", prepare_report_data_node)
    workflow.add_node("Supervisor", supervisor_chain)

    def entry_router(state: AgentState):
        if forced_next_list := state.get("next"):
            if forced_next_list and isinstance(forced_next_list, list) and len(forced_next_list) > 0:
                forced_next = forced_next_list[0]
                
                if forced_next == "VisualizationAgent":
                    state["next"] = ["VisualizationAgent"]
                    return "SQLAgent"
                
                if forced_next == "RAGAgent":
                    return "RAGSearchAgent"

                if forced_next in ["SQLAgent", "ReportAgent"]:
                    return forced_next

        return "Supervisor"

    workflow.add_conditional_edges(
        START,
        entry_router,
        {
            "Supervisor": "Supervisor",
            "RAGSearchAgent": "RAGSearchAgent",
            "SQLAgent": "SQLAgent",
            "ReportAgent": "ReportAgent",
            # VisualizationAgent is not a direct entry point
        },
    )

    def supervisor_router(
        state: AgentState,
    ) -> Literal[
        "SQLAgent",
        "RAGSearchAgent",
        "ReportAgent",
        "end",
    ]:
        next_routes = state.get("next")
        if not next_routes:
            return "end"

        next_route = next_routes.pop(0)
        state["next"] = next_routes
        
        valid_routes = [
            "SQLAgent",
            "RAGSearchAgent",
            "ReportAgent",
            "VisualizationAgent",
            "RAGCoachingAgent",
        ]
        if next_route not in valid_routes:
            return "end"
        return next_route

    workflow.add_conditional_edges(
        "Supervisor",
        supervisor_router,
        {
            "end": END,
            "SQLAgent": "SQLAgent",
            "RAGSearchAgent": "RAGSearchAgent",
            "ReportAgent": "ReportAgent",
            "VisualizationAgent": "VisualizationAgent",
            "RAGCoachingAgent": "RAGCoachingAgent",
        },
    )

    def after_sql_router(state: AgentState):
        if state.get("next"):
            next_agent = state["next"].pop(0)
            if next_agent == "ReportAgent":
                return "prepare_report_data"
            return next_agent
        return END

    workflow.add_conditional_edges(
        "SQLAgent",
        after_sql_router,
        {
            "prepare_report_data": "prepare_report_data",
            "VisualizationAgent": "VisualizationAgent",
            "RAGCoachingAgent": "RAGCoachingAgent",
            END: END,
        },
    )
    workflow.add_edge("prepare_report_data", "ReportAgent")

    return workflow.compile()

    return workflow.compile()
