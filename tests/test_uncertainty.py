"""Unit tests for UncertaintyQuantifier."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.uncertainty import UQResult, UncertaintyQuantifier, _entropy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_runnable(responses: list[str]) -> MagicMock:
    """Returns a Runnable whose ainvoke cycles through the given responses."""
    runnable = MagicMock()
    call_count = {"n": 0}

    async def _ainvoke(input, config=None):
        idx = call_count["n"] % len(responses)
        call_count["n"] += 1
        return responses[idx]

    runnable.ainvoke = _ainvoke
    return runnable


def _fake_judge(answers: dict[tuple[str, str], bool]) -> MagicMock:
    """Judge LLM that returns YES/NO based on a lookup dict."""
    judge = MagicMock()

    async def _ainvoke(messages, **kwargs):
        content = messages[0].content
        # Parse "Answer 1: X\n\nAnswer 2: Y\n\n..."
        lines = [l for l in content.splitlines() if l.strip()]
        a = lines[0].removeprefix("Answer 1: ").strip()
        b = lines[1].removeprefix("Answer 2: ").strip()
        agreed = answers.get((a, b), answers.get((b, a), False))
        reply = MagicMock()
        reply.content = "YES" if agreed else "NO"
        return reply

    judge.ainvoke = _ainvoke
    return judge


# ---------------------------------------------------------------------------
# _entropy
# ---------------------------------------------------------------------------

def test_entropy_all_agree():
    # One cluster of size M → entropy = 0
    assert _entropy([["a", "b", "c"]], 3) == pytest.approx(0.0)


def test_entropy_all_disagree():
    # M clusters of size 1 each → entropy = log(M)
    clusters = [["a"], ["b"], ["c"]]
    assert _entropy(clusters, 3) == pytest.approx(math.log(3))


def test_entropy_two_equal_clusters():
    clusters = [["a", "b"], ["c", "d"]]
    assert _entropy(clusters, 4) == pytest.approx(math.log(2))


# ---------------------------------------------------------------------------
# UncertaintyQuantifier
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_perfect_agreement():
    """M identical responses → 1 cluster, entropy 0."""
    responses = ["The AI Act regulates high-risk AI."] * 5
    runnable = _fake_runnable(responses)
    judge = _fake_judge({})  # never called when all go to first cluster

    uq = UncertaintyQuantifier(
        runnable=runnable,
        response_extractor=lambda r: r,
        judge_llm=judge,
        M=5,
    )
    result = await uq.quantify("What is the AI Act?")

    assert result.num_clusters == 1
    assert result.entropy == pytest.approx(0.0)
    assert len(result.responses) == 5


@pytest.mark.asyncio
async def test_total_disagreement():
    """M distinct responses that all disagree → M clusters, entropy = log(M)."""
    responses = [f"Answer {i}" for i in range(4)]
    runnable = _fake_runnable(responses)

    # All pairs disagree
    judge = _fake_judge({})

    uq = UncertaintyQuantifier(
        runnable=runnable,
        response_extractor=lambda r: r,
        judge_llm=judge,
        M=4,
    )
    result = await uq.quantify("ambiguous question")

    assert result.num_clusters == 4
    assert result.entropy == pytest.approx(math.log(4))


@pytest.mark.asyncio
async def test_two_clusters():
    """Responses split into two groups → 2 clusters, entropy = log(2)."""
    a = "AI Act covers high-risk AI."
    b = "AI Act bans all AI systems."

    responses = [a, a, b, b]
    runnable = _fake_runnable(responses)

    judge = _fake_judge({(a, a): True, (b, b): True, (a, b): False})

    uq = UncertaintyQuantifier(
        runnable=runnable,
        response_extractor=lambda r: r,
        judge_llm=judge,
        M=4,
    )
    result = await uq.quantify("Explain the AI Act")

    assert result.num_clusters == 2
    assert result.entropy == pytest.approx(math.log(2))


@pytest.mark.asyncio
async def test_for_llm_constructor():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))
    uq = UncertaintyQuantifier.for_llm(llm, judge_llm=llm, M=2)
    result = await uq.quantify([])
    assert isinstance(result, UQResult)
    assert result.num_clusters == 1


@pytest.mark.asyncio
async def test_for_graph_constructor():
    graph = MagicMock()
    graph.ainvoke = AsyncMock(
        return_value={"messages": [MagicMock(content="graph answer")]}
    )
    judge = MagicMock()
    judge.ainvoke = AsyncMock(return_value=MagicMock(content="YES"))

    uq = UncertaintyQuantifier.for_graph(graph, judge_llm=judge, M=2)
    result = await uq.quantify({"messages": [("human", "test")]})
    assert isinstance(result, UQResult)
