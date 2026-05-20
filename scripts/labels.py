"""canonical sentiment strings shared by gold + sklearn metrics."""

from __future__ import annotations

NEG = "negative"
NEU = "neutral"
POS = "positive"

# order for sklearn confusion_matrix / f1 labels=
LABEL_ORDER: tuple[str, ...] = (NEG, NEU, POS)

_ALIASES: dict[str, str] = {
    "neg": NEG,
    "negative": NEG,
    "neu": NEU,
    "neutral": NEU,
    "pos": POS,
    "positive": POS,
}


def canon(raw: str) -> str:
    """Strict label for human gold CSVs; raises if unknown."""
    s = raw.strip().lower()
    if s in _ALIASES:
        return _ALIASES[s]
    if s in LABEL_ORDER:
        return s
    raise ValueError(f"unknown sentiment label: {raw!r}")


def canon_pred_maybe(raw: str) -> str | None:
    """Model output: invalid/error/empty → None (counted as wrong / excluded from F1 per run_eval)."""
    s = raw.strip().lower()
    if s == "":
        return None
    # common failure tokens from LLM exports
    if s in {"error", "invalid", "none", "nan", "null", "unknown", "n/a", "na"}:
        return None
    if s in _ALIASES:
        return _ALIASES[s]
    if s in LABEL_ORDER:
        return s
    return None
