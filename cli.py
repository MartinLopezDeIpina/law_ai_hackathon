"""Interactive CLI for the EU law agent."""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessageChunk, HumanMessage

from app.agents.graphs.simple_react_graph.graph import build_graph

_THREAD_CONFIG = {"configurable": {"thread_id": "cli-1"}}


async def main() -> None:
    graph = build_graph()
    print("EU Law Assistant  (type 'exit' to quit)\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() in ("exit", "quit"):
            break
        print("Assistant: ", end="", flush=True)
        async for msg, meta in graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=_THREAD_CONFIG,
            stream_mode="messages",
        ):
            if isinstance(msg, AIMessageChunk) and meta["langgraph_node"] == "agent":
                print(msg.content, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
