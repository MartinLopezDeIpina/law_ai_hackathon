"""Simple ReAct agent graph backed by a configurable LLM and the RAG/CELLAR tools."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.agents.tools import TOOLS
from app.core.llm import get_llm

_SYSTEM_PROMPT = (
    "You are an expert EU law assistant. "
    "Use rag_search to look up EU legislation from the local knowledge base before answering. "
    "Cite CELEX identifiers when referencing specific acts."
)


def build_graph():
    llm = get_llm()
    checkpointer = MemorySaver()
    return create_react_agent(
        llm,
        tools=TOOLS,
        prompt=_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
