import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from agent.nodes import validate_dataset, process_dataset, train_model, generate_suggestions
from agent.state import State

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

builder = StateGraph(State)

builder.add_node("validate",validate_dataset)
builder.add_node("process",process_dataset)
builder.add_node("train", train_model)
builder.add_node("generate",generate_suggestions)

builder.add_edge(START, "validate")
builder.add_edge("validate", "process")
builder.add_edge("validate", "process")
builder.add_edge("process", "train")
builder.add_edge("train", "generate")
builder.add_edge("generate", END)

memory = MemorySaver()
graph = builder.compile()


