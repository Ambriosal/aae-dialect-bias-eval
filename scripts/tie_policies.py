"""how to break 3-way disagreements on the common 100. swap impl or pass a custom one."""

from __future__ import annotations

from typing import Protocol


class TiePolicy(Protocol):
    def pick(self, a1: str, a2: str, a3: str) -> str: ...


class AnnotatorOne:
    """ryan/common case: if all three sentiments differ, trust annotator 1."""

    def pick(self, a1: str, a2: str, a3: str) -> str:
        return a1


class AnnotatorTwo:
    def pick(self, a1: str, a2: str, a3: str) -> str:
        return a2


class AnnotatorThree:
    def pick(self, a1: str, a2: str, a3: str) -> str:
        return a3


def default_tie_policy() -> TiePolicy:
    return AnnotatorOne()
