from typing import Annotated, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from pandas import DataFrame


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    dataset: DataFrame
    valid: bool
    error: str
    results: Dict[str, Any]