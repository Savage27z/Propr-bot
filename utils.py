"""Shared helpers: MarkdownV2 escaping, number formatting.

These used to live in :mod:`handlers`. They're factored out so
:mod:`pnl` and :mod:`ws_listener` can reuse them without import cycles.
The implementations are identical to the originals so every existing
reply keeps working byte-for-byte.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

MARKDOWN_V2_ESCAPE = r"_*[]()~`>#+-=|{}.!\\"


def escape_md(text: Any) -> str:
    """Escape MarkdownV2 special characters in *text*.

    Pass any user-supplied or API-derived value through this before embedding
    it into a reply that uses ``ParseMode.MARKDOWN_V2``.
    """
    s = "" if text is None else str(text)
    out = []
    for ch in s:
        if ch in MARKDOWN_V2_ESCAPE:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def fmt_num(value: Any, places: int = 2) -> str:
    """Format *value* as a decimal with up to *places* digits after the point.

    Trailing zeros are stripped. Non-numeric input is returned as-is.
    """
    if value is None or value == "":
        return "-"
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return str(value)
    quantizer = Decimal("1").scaleb(-places) if places > 0 else Decimal("1")
    try:
        d = d.quantize(quantizer)
    except InvalidOperation:
        pass
    text = format(d, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"
