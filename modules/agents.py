# modules/agents.py

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import AIMessage

from .tools import get_rag_tool

def create_agent(llm, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [AIMessage(content=result["output"], name=name)],
        "sender": name,
    }

def create_rag_agent_executor(llm):
    system_prompt = "You are an expert at retrieving information from PDF documents. Use the `pdf_document_retriever` tool to find answers."
    rag_tool = get_rag_tool()
    if not rag_tool:
        return create_agent(llm, [], "The document retrieval tool is not available.")
    return create_agent(llm, [rag_tool], system_prompt)

def direct_response_node(state, llm):
    response = llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=response.content, name="DirectResponse")]}