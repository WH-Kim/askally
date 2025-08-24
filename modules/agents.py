# modules/agents.py

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from .prompts import (
    CHART_GENERATOR_PROMPT,
    DIRECT_RESPONSE_PROMPT,
    RAG_AGENT_PROMPT,
    REPORT_AGENT_PROMPT,
    SQL_AGENT_PROMPT,
)
from .state import AgentState
from .tools import python_repl, create_pdf_report, execute_sql_and_get_results, get_schema, list_tables

def text_to_sql_query(messages: list, llm: Runnable, table_info: str, few_shot_examples: list) -> str:
    examples_str = "\n\n".join([f"User question: {ex['question']}\nSQL query: {ex['query']}" for ex in few_shot_examples])
    system_prompt = f"""You are an expert in converting natural language questions into SQL queries. Based on the user's question and the database schema provided, generate a syntactically correct SQL query.

**Database Schema:**
{table_info}

**Few-shot Examples:**
{examples_str}

**Rules:**
- Only generate the SQL query. Do not add any other text or explanation."""
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{question}")])
    chain = prompt | llm
    
    # 메시지 리스트에서 마지막 HumanMessage를 찾아 질문으로 사용
    user_question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    response = chain.invoke({"question": user_question})
    # 생성된 SQL 쿼리에서 불필요한 마크다운 제거
    return response.content.strip().replace("```sql", "").replace("```", "").strip()

def direct_response_node(state: AgentState, llm: Runnable):
    prompt = ChatPromptTemplate.from_messages([("system", DIRECT_RESPONSE_PROMPT), MessagesPlaceholder(variable_name="messages")])
    chain = prompt | llm
    for chunk in chain.stream(state):
        yield {"messages": [chunk]}

def rag_agent_node(state: AgentState, llm: Runnable, retriever: Runnable):
    if not retriever:
        yield {"messages": [AIMessageChunk(content="RAG 에이전트를 사용할 수 없습니다. 벡터 DB가 초기화되지 않았습니다.")]}
        return
        
    # 마지막 사용자 질문을 기반으로 문서 검색
    last_user_question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_question = msg.content
            break
            
    retrieved_docs = retriever.invoke(last_user_question)
    context = "\n\n---\n\n".join([f"## 문서 제목: {doc.metadata.get('source', 'N/A')}\n\n" + doc.page_content for doc in retrieved_docs])
    prompt = ChatPromptTemplate.from_messages([("system", RAG_AGENT_PROMPT), MessagesPlaceholder(variable_name="messages"), ("system", "--- 검색된 문서 내용 ---\n\n{context}")])
    chain = prompt | llm
    for chunk in chain.stream({"messages": state["messages"], "context": context}):
        yield {"messages": [chunk]}

def chart_generator_agent_node(state: AgentState, llm: Runnable, db_info: dict):
    conversation_history = state["messages"]
    last_user_message = conversation_history[-1].content
    
    # DB 관련 키워드가 있을 경우에만 SQL 쿼리 실행
    if any(k in last_user_message.lower() for k in ['db', '데이터', '고객']):
        all_tables = list_tables.invoke({})
        schema = get_schema.invoke({"table_names": ",".join(all_tables)})
        sql_query = text_to_sql_query([conversation_history[-1]], llm, schema, db_info.get("few_shot_examples", []))
        sql_result_json = execute_sql_and_get_results.invoke({"query": sql_query})
        sql_tool_message = ToolMessage(content=sql_result_json, tool_call_id="sql_call")
        sql_ai_message = AIMessage(content="", tool_calls=[{"name": "sql_query_tool", "args": {"query": sql_query}, "id": "sql_call"}])
        yield {"messages": [sql_ai_message, sql_tool_message]}
        messages_for_code_gen = conversation_history + [sql_ai_message, sql_tool_message]
        user_prompt_context = f"쿼리 실행 결과:\n```json\n{sql_result_json}\n```\n위 데이터를 바탕으로 차트 생성 코드를 작성해주세요."
    else:
        messages_for_code_gen = conversation_history
        conversation_summary = "\n".join([f"{msg.type}: {msg.content}" for msg in conversation_history])
        user_prompt_context = f"다음 대화 내용을 바탕으로, 내용을 요약하고 핵심 인사이트를 보여주는 차트 생성 코드를 작성해주세요.\n\n대화 내용:\n{conversation_summary}"

    code_gen_prompt = ChatPromptTemplate.from_messages([("system", CHART_GENERATOR_PROMPT), MessagesPlaceholder(variable_name="messages"), ("user", "{context}")])
    code_gen_chain = code_gen_prompt | llm.bind_tools([python_repl])
    
    response_with_tool_call = code_gen_chain.invoke({"messages": messages_for_code_gen, "context": user_prompt_context})
    yield {"messages": [response_with_tool_call]}

    if response_with_tool_call.tool_calls:
        tool_call = response_with_tool_call.tool_calls[0]
        tool_output = python_repl.invoke(tool_call["args"])
        python_tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
        yield {"messages": [python_tool_message]}

        final_prompt = ChatPromptTemplate.from_messages([("system", CHART_GENERATOR_PROMPT), MessagesPlaceholder(variable_name="messages"), ("user", "차트 생성 결과를 바탕으로 최종 답변을 생성하고, 답변에 이미지 파일 경로를 포함하세요.")])
        final_chain = final_prompt | llm
        messages_for_final_step = messages_for_code_gen + [response_with_tool_call, python_tool_message]
        for chunk in final_chain.stream({"messages": messages_for_final_step}):
            yield {"messages": [chunk]}
    else:
        # 도구 호출 없이 텍스트만 반환된 경우, 해당 내용을 스트리밍
        for char in list(response_with_tool_call.content):
            yield {"messages": [AIMessageChunk(content=char)]}

# [수정] Report Agent 로직 개선
def report_agent_node(state: AgentState, llm: Runnable):
    """
    대화 내용을 요약하여 PDF 보고서를 생성하는 노드.
    1. LLM을 사용하여 대화 내용 요약 및 보고서 콘텐츠 생성
    2. 생성된 콘텐츠로 create_pdf_report 도구 호출
    3. 사용자에게 보고서 파일 경로를 포함한 최종 답변 생성 및 스트리밍
    """
    # 1. 보고서 콘텐츠 생성
    conversation_text = "\n".join([f"**{msg.type.upper()}**: {msg.content}" for msg in state['messages'] if msg.content])
    
    # 보고서 제목과 내용을 생성하기 위한 프롬프트
    report_content_prompt = ChatPromptTemplate.from_messages([
        ("system", REPORT_AGENT_PROMPT + "\n\n먼저 대화 내용을 바탕으로 보고서의 제목과 내용을 생성하세요. '제목:'과 '내용:'으로 구분하여 작성해주세요."),
        ("user", "다음 대화 내용을 요약하여 보고서의 제목과 내용을 만들어주세요.\n\n**대화 내용:**\n{conversation}")
    ])
    
    # LLM을 호출하여 보고서 제목과 내용 생성
    report_creation_chain = report_content_prompt | llm
    generated_report_text = report_creation_chain.invoke({"conversation": conversation_text}).content

    # 생성된 텍스트에서 제목과 내용 분리
    try:
        title = generated_report_text.split("제목:")[1].split("내용:")[0].strip()
        content = generated_report_text.split("내용:")[1].strip()
    except IndexError:
        title = "대화 요약 보고서"
        content = generated_report_text

    # 2. PDF 생성 도구 호출
    # AIMessage를 추가하여 어떤 도구를 호출했는지 기록
    yield {"messages": [AIMessage(content=f"'{title}' 제목으로 PDF 보고서 생성을 시작합니다.")]}
    
    # 도구 직접 호출
    try:
        tool_output = create_pdf_report.invoke({"title": title, "content": content})
        report_path = str(tool_output)
        # ToolMessage를 추가하여 도구 실행 결과 기록
        report_tool_message = ToolMessage(content=report_path, tool_call_id="create_pdf_report_call")
        yield {"messages": [report_tool_message]}
    except Exception as e:
        error_message = f"PDF 생성 중 오류가 발생했습니다: {e}"
        yield {"messages": [AIMessage(content=error_message)]}
        return

    # 3. 사용자에게 최종 답변 생성 및 스트리밍
    final_message_content = f"요청하신 대화 내용에 대한 PDF 보고서가 생성되었습니다. 아래 버튼을 통해 다운로드하실 수 있습니다.\n\n파일 경로: {report_path}"
    
    # 스트리밍 효과를 위해 한 글자씩 yield
    for char in list(final_message_content):
        yield {"messages": [AIMessageChunk(content=char)]}