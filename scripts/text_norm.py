"""one string key for joining gold rows to big model csvs."""

from __future__ import annotations

import re
import unicodedata


_ws = re.compile(r"\s+")


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s.strip())
    s = _ws.sub(" ", s)
    return s
