"""Uncertainty quantification via semantic clustering and entropy.

Samples a LangChain Runnable (raw LLM or LangGraph graph) M times, clusters
the responses by semantic agreement using a judge LLM, then computes entropy
over the cluster probability distribution.

Usage::

    uq = UncertaintyQuantifier.for_graph(graph, judge_llm=get_llm(), M=10)
    result = await uq.quantify({"messages": [("human", "What is the AI Act?")]})
    print(result.entropy, result.num_clusters)
"""

from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langsmith import traceable

_JUDGE_PROMPT = """\
Answer 1: {a}

Answer 2: {b}

Do these two answers express the same conclusion or factual claim? \
Reply with YES or NO only."""


@dataclass
class UQResult:
    entropy: float
    num_clusters: int
    clusters: list[list[str]]
    responses: list[str]


class UncertaintyQuantifier:
    def __init__(
        self,
        runnable: Runnable,
        response_extractor: Callable[[Any], str],
        judge_llm: BaseChatModel,
        M: int = 10,
    ) -> None:
        self.runnable = runnable
        self.extractor = response_extractor
        self.judge_llm = judge_llm
        self.M = M

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def for_llm(
        cls,
        llm: BaseChatModel,
        judge_llm: BaseChatModel | None = None,
        M: int = 10,
    ) -> "UncertaintyQuantifier":
        return cls(
            runnable=llm,
            response_extractor=lambda r: r.content,
            judge_llm=judge_llm or llm,
            M=M,
        )

    @classmethod
    def for_graph(
        cls,
        graph: Runnable,
        judge_llm: BaseChatModel | None = None,
        M: int = 10,
        llm: BaseChatModel | None = None,
    ) -> "UncertaintyQuantifier":
        if judge_llm is None and llm is None:
            raise ValueError("Provide judge_llm (or llm as a fallback) for for_graph()")
        return cls(
            runnable=graph,
            response_extractor=lambda r: r["messages"][-1].content,
            judge_llm=judge_llm or llm,  # type: ignore[arg-type]
            M=M,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @traceable(name="uncertainty_quantify")
    async def quantify(self, input: Any) -> UQResult:
        responses = await self._sample(input)
        clusters = await self._cluster(responses)
        h = _entropy(clusters, self.M)
        return UQResult(
            entropy=h,
            num_clusters=len(clusters),
            clusters=clusters,
            responses=responses,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _sample(self, input: Any) -> list[str]:
        async def _one(i: int) -> str:
            # Give each sample its own thread_id so stateful graphs don't
            # share checkpoint state across samples.
            config = {"configurable": {"thread_id": f"uq-{uuid.uuid4()}"}}
            raw = await self.runnable.ainvoke(input, config=config)
            return self.extractor(raw)

        return list(await asyncio.gather(*(_one(i) for i in range(self.M))))

    async def _cluster(self, responses: list[str]) -> list[list[str]]:
        clusters: list[list[str]] = []
        for response in responses:
            placed = False
            for cluster in clusters:
                if await self._agrees(response, cluster[0]):
                    cluster.append(response)
                    placed = True
                    break
            if not placed:
                clusters.append([response])
        return clusters

    async def _agrees(self, a: str, b: str) -> bool:
        if a == b:
            return True
        prompt = _JUDGE_PROMPT.format(a=a, b=b)
        reply = await self.judge_llm.ainvoke([HumanMessage(content=prompt)])
        return reply.content.strip().upper().startswith("YES")


def _entropy(clusters: list[list[str]], M: int) -> float:
    probs = [len(c) / M for c in clusters]
    return -sum(p * math.log(p) for p in probs if p > 0)
