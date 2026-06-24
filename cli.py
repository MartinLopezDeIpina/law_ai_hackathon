"""Interactive CLI for the EU law agent.

Usage:
    python cli.py                      # chat + uncertainty score (UQ on by default)
    python cli.py --no-uq              # plain chat, no UQ
    python cli.py --uq-samples 8       # use 8 samples instead of the default 5
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessageChunk, HumanMessage

from app.agents.graphs.simple_react_graph.graph import build_graph
from app.agents.uncertainty import UncertaintyQuantifier
from app.core.llm import get_llm

_THREAD_CONFIG = {"configurable": {"thread_id": "cli-1"}}


async def main() -> None:
    parser = argparse.ArgumentParser(description="EU Law Assistant CLI")
    parser.add_argument("--no-uq", dest="uq", action="store_false", help="Disable uncertainty quantification")
    parser.add_argument("--uq-samples", type=int, default=5, metavar="N", help="Number of UQ samples (default: 5)")
    args = parser.parse_args()

    graph = build_graph()
    uq = UncertaintyQuantifier.for_graph(graph, judge_llm=get_llm(), M=args.uq_samples) if args.uq else None

    print("EU Law Assistant  (type 'exit' to quit)\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() in ("exit", "quit"):
            break

        # Start UQ sampling concurrently so it runs while the user reads the answer.
        uq_task = (
            asyncio.create_task(uq.quantify({"messages": [HumanMessage(content=user_input)]}))
            if uq else None
        )

        print("Assistant: ", end="", flush=True)
        async for msg, meta in graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=_THREAD_CONFIG,
            stream_mode="messages",
        ):
            if isinstance(msg, AIMessageChunk) and meta["langgraph_node"] == "agent":
                print(msg.content, end="", flush=True)
        print("\n")

        if uq_task:
            result = await uq_task
            print(f"[UQ] entropy={result.entropy:.3f}  clusters={result.num_clusters}/{args.uq_samples}\n")


if __name__ == "__main__":
    asyncio.run(main())
