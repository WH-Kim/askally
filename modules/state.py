# modules/state.py

from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    sender: str
    query: Optional[str]
    result: Optional[str]
    schema: Optional[str]
    table_names: Optional[List[str]]