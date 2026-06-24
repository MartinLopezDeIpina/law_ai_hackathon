from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Concept:
    uri: str | None
    label: str | None


@dataclass
class Agent:
    uri: str | None
    label: str | None


@dataclass
class Work:
    uri: str
    celex: str | None = None
    title: str | None = None
    date: str | None = None
    abstract: str | None = None
    text: str | None = None


@dataclass
class ExpandedWork:
    uri: str
    celex: str | None
    title: str | None
    date: str | None
    abstract: str | None
    text: str | None
    score: float
    concepts: list[Concept] = field(default_factory=list)
    citations: list[Work] = field(default_factory=list)
    agents: list[Agent] = field(default_factory=list)
