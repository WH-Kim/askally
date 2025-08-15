from typing import List
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool


def create_agent_executor(model_name: str, tools: List[BaseTool]):
    """Create a supervisor agent that can use provided tools.

    Args:
        model_name (str): OpenAI model name.
        tools (List[BaseTool]): Tools available to the agent.

    Returns:
        Any: A LangGraph agent executor capable of handling the tools.
    """
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    checkpointer = MemorySaver()
    agent_executor = create_react_agent(llm, tools=tools, checkpointer=checkpointer)
    return agent_executor
