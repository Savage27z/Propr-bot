"""PnL snapshot, text formatting, and image-card rendering.

Self-contained module: does all PnL-related work (computation, MarkdownV2
text, PNG card) so that :mod:`handlers` and :mod:`ws_listener` can keep
their own concerns clean. Only depends on :mod:`utils` for shared helpers;
must never import from :mod:`handlers` (cycle risk).

All math uses :class:`decimal.Decimal` \u2014 numeric fields from the Propr API
come as strings, so we parse once via :func:`_dec` and stay in ``Decimal``.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from utils import escape_md as _default_escape_md
from utils import fmt_num

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PositionPnL:
    """A single position's PnL fields, all parsed into :class:`Decimal`."""

    asset: str
    side: str          # "long" | "short"
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    leverage: Decimal
    margin_used: Decimal
    roe_pct: Decimal   # returnOnEquity * 100


@dataclass
class PnLSnapshot:
    """Aggregated PnL view for all open positions at a point in time."""

    positions: List[PositionPnL]
    total_unrealized: Decimal
    total_realized: Decimal
    total_margin_used: Decimal
    generated_at: datetime  # UTC

    @property
    def total(self) -> Decimal:
        return self.total_unrealized + self.total_realized


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _dec(value: Any, default: str = "0") -> Decimal:
    """Coerce *value* to :class:`Decimal`; return ``Decimal(default)`` on failure."""
    if value is None or value == "":
        return Decimal(default)
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(default)


def _signed(value: Decimal, places: int = 2) -> str:
    """Render *value* with a leading ``+``/``-`` (bare ``0`` for zero).

    Uses :func:`fmt_num` so trailing zeros are stripped.
    """
    text = fmt_num(value, places)
    if text == "0" or text.startswith("-"):
        return text
    return "+" + text


def _asset_symbol(raw: str) -> str:
    """Strip any ``xyz:`` prefix and upper-case the symbol."""
    if not raw:
        return "?"
    return str(raw).split(":")[-1].upper() or "?"


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------
def build_snapshot(raw_positions: Iterable[dict]) -> PnLSnapshot:
    """Turn the raw Propr position list into a :class:`PnLSnapshot`.

    Positions with ``quantity == 0`` are filtered out; totals sum across the
    remaining rows.
    """
    positions: List[PositionPnL] = []
    for raw in raw_positions or []:
        if not isinstance(raw, dict):
            continue
        quantity = _dec(raw.get("quantity", raw.get("qty")))
        if quantity == 0:
            continue
        side = str(raw.get("positionSide") or raw.get("side") or "").lower()
        roe = _dec(raw.get("returnOnEquity"))
        positions.append(
            PositionPnL(
                asset=_asset_symbol(raw.get("asset", "")),
                side=side,
                quantity=quantity,
                entry_price=_dec(raw.get("entryPrice", raw.get("avgEntryPrice"))),
                mark_price=_dec(
                    raw.get("markPrice", raw.get("mark", raw.get("price")))
                ),
                unrealized_pnl=_dec(raw.get("unrealizedPnl", raw.get("uPnl"))),
                realized_pnl=_dec(raw.get("realizedPnl", raw.get("rPnl"))),
                leverage=_dec(raw.get("leverage"), "1"),
                margin_used=_dec(raw.get("marginUsed", raw.get("margin"))),
                roe_pct=roe * Decimal(100),
            )
        )

    total_u = sum((p.unrealized_pnl for p in positions), Decimal(0))
    total_r = sum((p.realized_pnl for p in positions), Decimal(0))
    total_m = sum((p.margin_used for p in positions), Decimal(0))
    return PnLSnapshot(
        positions=positions,
        total_unrealized=total_u,
        total_realized=total_r,
        total_margin_used=total_m,
        generated_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# MarkdownV2 formatter
# ---------------------------------------------------------------------------
_SEPARATOR = "────────────────────────"  # light box-drawing — no MD2 escape needed


def format_snapshot_markdown(snapshot: PnLSnapshot, escape_md=None) -> str:
    """Return a MarkdownV2 string ready to send to Telegram.

    All dynamic content is pushed through *escape_md* (defaulting to the
    shared :func:`utils.escape_md`); static punctuation is pre-escaped in
    the string literals below. ``|``/``(``/``)``/``+``/``.`` \u2014 the MD2
    specials that matter for this layout \u2014 are either escaped inline or
    come out of ``escape_md``.
    """
    esc = escape_md or _default_escape_md

    if not snapshot.positions:
        return "💭 No open positions\\."

    lines: List[str] = ["💰 *PnL Snapshot*", _SEPARATOR]

    for pos in snapshot.positions:
        side_emoji = "📈" if pos.side == "long" else "📉"
        side_text = pos.side.upper() if pos.side else "?"

        qty_s = fmt_num(pos.quantity, 6)
        entry_s = fmt_num(pos.entry_price)
        mark_s = fmt_num(pos.mark_price)

        upnl_str = _signed(pos.unrealized_pnl)
        notional = pos.quantity * pos.entry_price
        if notional != 0:
            pnl_pct = (pos.unrealized_pnl / notional) * Decimal(100)
        else:
            pnl_pct = Decimal(0)
        pnl_pct_str = _signed(pnl_pct)
        dot = "🟢" if pos.unrealized_pnl >= 0 else "🔴"

        rpnl_s = fmt_num(pos.realized_pnl)
        lev_s = fmt_num(pos.leverage, 0)

        # Header line: emoji, side, asset, qty @ entry -> mark (latter in code spans)
        lines.append(
            f"{side_emoji} {esc(side_text)}  {esc(pos.asset)}  "
            f"`{qty_s} @ {entry_s}` → `{mark_s}`"
        )
        # uPnL line
        lines.append(
            f"   uPnL: *{esc(upnl_str)} USDC* "
            f"\\({esc(pnl_pct_str)}%\\)  {dot}"
        )
        # rPnL + leverage
        lines.append(
            f"   rPnL: {esc(rpnl_s)} USDC  \\|  lev {esc(lev_s)}x"
        )

    lines.append(_SEPARATOR)

    total_dot = "🟢" if snapshot.total_unrealized >= 0 else "🔴"
    lines.append(
        f"Total uPnL: *{esc(_signed(snapshot.total_unrealized))} USDC*  {total_dot}"
    )
    lines.append(
        f"Total rPnL: `{fmt_num(snapshot.total_realized)} USDC`"
    )
    lines.append(
        f"*Combined: {esc(_signed(snapshot.total))} USDC*"
    )
    lines.append(
        f"Margin used: `{fmt_num(snapshot.total_margin_used)} USDC`"
    )
    ts = snapshot.generated_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append(f"📅 {esc(ts)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PNG share-card rendering (Propr-style, 1200x900, one position per card)
# ---------------------------------------------------------------------------
_CARD_W, _CARD_H = 1200, 900

# Fallback solid background when ``assets/bg.jpg`` is missing at render time.
_FALLBACK_BG = (13, 17, 23)  # #0D1117

# Tint overlays mixed with the blurred background at alpha 0.45.
_TINT_PROFIT = (20, 90, 55)
_TINT_LOSS = (90, 30, 30)
_TINT_NEUTRAL = (25, 45, 35)

# Text colors.
_COL_WHITE = (255, 255, 255)
_COL_DIM = (201, 209, 217)           # #C9D1D9
_COL_GRAY = (125, 133, 144)          # #7D8590
_COL_PROFIT = (74, 222, 128)         # #4ADE80
_COL_LOSS = (248, 113, 113)          # #F87171

# Per-asset brand colors for the icon disc.
_ASSET_COLORS = {
    "BTC": "#F7931A",
    "ETH": "#627EEA",
    "SOL": "#14F195",
    "DOGE": "#C2A633",
    "XRP": "#23292F",
    "AVAX": "#E84142",
    "LINK": "#2A5ADA",
}
_ASSET_FALLBACK_COLOR = "#6B7280"

# Background asset. Kept relative to this module so the bot works regardless of
# the caller's CWD. File is committed to the repo \u2014 see task spec.
_BG_PATH = os.path.join(os.path.dirname(__file__), "assets", "bg.jpg")

_FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT_PATH_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

_FONT_FALLBACK_WARNED = False
# review finding 5: module-level flag so the bg-missing warning fires once
# per process instead of spamming on every rendered card.
_BG_WARNED = False


def _font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    """Return a TrueType font at *size*, falling back to PIL default once.

    Tries DejaVuSans-Bold first (or DejaVuSans when ``bold=False``) then the
    other, then ``ImageFont.load_default()`` with a one-shot warning so we
    don't flood logs on every card.
    """
    global _FONT_FALLBACK_WARNED
    candidates: Tuple[str, ...] = (
        (_FONT_PATH_BOLD, _FONT_PATH_REG) if bold else (_FONT_PATH_REG, _FONT_PATH_BOLD)
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except (OSError, IOError):
            continue
    if not _FONT_FALLBACK_WARNED:
        log.warning(
            "DejaVu fonts not found, falling back to PIL default (lower quality)"
        )
        _FONT_FALLBACK_WARNED = True
    return ImageFont.load_default()


def _fmt_signed(x: Decimal, places: int = 1) -> str:
    """Format *x* with a mandatory sign to *places* decimals (``+6.5``, ``-4.2``)."""
    q = Decimal(10) ** -places if places > 0 else Decimal(1)
    try:
        d = x.quantize(q)
    except InvalidOperation:
        d = x
    sign = "-" if d < 0 else "+"
    abs_text = format(abs(d), "f")
    if places > 0 and "." not in abs_text:
        abs_text = f"{abs_text}.{'0' * places}"
    return sign + abs_text


def _fmt_price(x: Decimal) -> str:
    """Format a price with thousands separator and 2 decimal places."""
    try:
        d = x.quantize(Decimal("0.01"))
    except InvalidOperation:
        d = x
    # Python's ``,`` format spec on Decimal gives us the grouping.
    return f"{d:,.2f}"


def _text_wh(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
) -> Tuple[int, int, int, int]:
    """Return ``(width, height, x_offset, y_offset)`` from ``textbbox``."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1], bbox[0], bbox[1]


def _asset_icon(asset: str, size: int) -> Image.Image:
    """Render the round asset icon: colored disc + first-letter glyph."""
    color = _ASSET_COLORS.get(asset.upper(), _ASSET_FALLBACK_COLOR)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((0, 0, size - 1, size - 1), fill=color)

    letter = (asset[:1] or "?").upper()
    # ~70% of the disc is a typographically pleasing glyph size.
    glyph_font = _font(max(12, int(size * 0.7)), bold=True)
    w, h, ox, oy = _text_wh(draw, letter, glyph_font)
    draw.text(
        ((size - w) / 2 - ox, (size - h) / 2 - oy),
        letter,
        font=glyph_font,
        fill=_COL_WHITE,
    )
    return img


def _propr_mark(size: int) -> Image.Image:
    """Render the stylized Propr mark (rounded white square + black X).

    Returns an RGBA image of *size* pixels square ready to be pasted onto
    the card background.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    radius = max(2, size // 6)
    draw.rounded_rectangle(
        (0, 0, size - 1, size - 1),
        radius=radius,
        fill=(255, 255, 255, 235),
    )
    pad = size // 6
    stroke = max(2, size // 18)
    draw.line((pad, pad, size - pad, size - pad), fill=(0, 0, 0, 255), width=stroke)
    draw.line((size - pad, pad, pad, size - pad), fill=(0, 0, 0, 255), width=stroke)
    return img


def _apply_bg(path: str, w: int, h: int, is_profit: bool, empty: bool) -> Image.Image:
    """Build the base card image: blurred background + tint overlay.

    Falls back to a solid dark color if the JPEG is missing or fails to load.
    The *empty* flag uses the neutral (cool green) tint so the empty-state
    card doesn't misrepresent account state.

    Review finding 5 hardens this further: on any ``Image.open`` failure (or
    a missing file) we emit a single module-level warning via ``_BG_WARNED``
    and return the tinted fallback fill rather than raising into the caller.
    """
    global _BG_WARNED
    base: Optional[Image.Image] = None
    # review finding 5: guard Image.open so a broken/missing bg.jpg degrades
    # to the fallback fill without propagating a Pillow exception to pnl.py
    # callers; log once per process via _BG_WARNED.
    try:
        raw = Image.open(path).convert("RGB")
        base = raw.resize((w, h), Image.LANCZOS)
        base = base.filter(ImageFilter.GaussianBlur(radius=18))
        base = ImageEnhance.Brightness(base).enhance(0.35)
    except Exception as exc:  # noqa: BLE001 — any failure degrades to fallback fill
        if not _BG_WARNED:
            log.warning("assets/bg.jpg unavailable (%s); using fallback fill", exc)
            _BG_WARNED = True
        base = None

    if base is None:
        base = Image.new("RGB", (w, h), _FALLBACK_BG)

    if empty:
        tint_color = _TINT_NEUTRAL
    else:
        tint_color = _TINT_PROFIT if is_profit else _TINT_LOSS
    tint = Image.new("RGB", (w, h), tint_color)
    return Image.blend(base, tint, 0.45)


def _draw_bottom_gradient(
    draw: ImageDraw.ImageDraw, w: int, h: int, height: int = 200
) -> None:
    """Darken the bottom *height* px progressively toward full black.

    Implemented with a stack of 1px-tall rectangles whose alpha ramps from
    0 at the top of the band to ~220 at the very bottom, giving the entry /
    mark row extra legibility without a hard edge.
    """
    # The draw target is RGB so we draw with an RGBA source onto an overlay.
    overlay_img = draw._image  # type: ignore[attr-defined]
    overlay = Image.new("RGBA", (w, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for i in range(height):
        # Ease-in: quadratic ramp looks smoother than linear on dark imagery.
        t = (i / max(1, height - 1)) ** 2
        alpha = int(220 * t)
        overlay_draw.line([(0, i), (w, i)], fill=(0, 0, 0, alpha))
    overlay_img.paste(overlay, (0, h - height), overlay)


def _pick_focus(snapshot: PnLSnapshot) -> Optional[PositionPnL]:
    """Select the position with the largest absolute unrealized PnL."""
    if not snapshot.positions:
        return None
    return max(snapshot.positions, key=lambda p: abs(p.unrealized_pnl))


def render_snapshot_image(
    snap: PnLSnapshot, *, focus: Optional[PositionPnL] = None
) -> bytes:
    """Render a single-position share card as PNG bytes.

    If *focus* is provided, render that position. Otherwise pick the position
    with the largest absolute unrealized PnL. If ``snap.positions`` is empty,
    render a neutral 'No open positions' card (same layout, dimmed).
    """
    w, h = _CARD_W, _CARD_H

    if focus is None:
        focus = _pick_focus(snap)

    empty = focus is None
    is_profit = bool(focus and focus.unrealized_pnl >= 0)

    img = _apply_bg(_BG_PATH, w, h, is_profit=is_profit, empty=empty)
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    _draw_bottom_gradient(draw, w, h, height=200)

    # --- Top-left: asset icon + ticker + side/leverage -------------------
    if focus is not None:
        icon_size = 72
        icon = _asset_icon(focus.asset, icon_size)
        img.paste(icon, (60, 60 - 20), icon)  # center the icon around y=60..132

        ticker_font = _font(64, bold=True)
        ticker = focus.asset
        t_w, t_h, t_ox, t_oy = _text_wh(draw, ticker, ticker_font)
        ticker_x = 60 + icon_size + 20
        # Baseline alignment: visual midline of the icon is at ~y=76.
        ticker_y = 52 - t_oy
        draw.text((ticker_x - t_ox, ticker_y), ticker, font=ticker_font, fill=_COL_WHITE)

        lev_int = int(focus.leverage) if focus.leverage else 1
        side_label = (focus.side or "?").capitalize()
        side_text = f"{side_label} {lev_int}x"
        side_font = _font(42, bold=True)
        s_w, s_h, s_ox, s_oy = _text_wh(draw, side_text, side_font)
        side_x = ticker_x + t_w + 30
        # Baseline-aligned with the ticker's visual midline.
        side_y = 52 - s_oy + (t_h - s_h) // 2
        draw.text((side_x - s_ox, side_y), side_text, font=side_font, fill=_COL_DIM)
    else:
        # Dim neutral header when no position \u2014 just a plain PROPR mark kept
        # intact on the right (below).
        pass

    # --- Top-right: Propr mark + PROPR wordmark --------------------------
    propr_size = 72
    propr = _propr_mark(propr_size)
    word_font = _font(56, bold=True)
    word_w, word_h, word_ox, word_oy = _text_wh(draw, "PROPR", word_font)

    gap = 20
    right_margin = 60
    total_w = propr_size + gap + word_w
    propr_x = w - right_margin - total_w
    propr_y = 60 - 20  # align with top-left icon band
    img.paste(propr, (propr_x, propr_y), propr)
    draw.text(
        (propr_x + propr_size + gap - word_ox, 52 - word_oy),
        "PROPR",
        font=word_font,
        fill=_COL_WHITE,
    )

    # --- Headline ROI % + secondary PnL line -----------------------------
    if focus is not None:
        if focus.margin_used == 0:
            roi_pct = Decimal(0)
        else:
            roi_pct = (focus.unrealized_pnl / focus.margin_used) * Decimal(100)
        pnl_color = _COL_PROFIT if focus.unrealized_pnl >= 0 else _COL_LOSS

        roi_text = _fmt_signed(roi_pct, 1) + "%"
        roi_font = _font(220, bold=True)
        r_w, r_h, r_ox, r_oy = _text_wh(draw, roi_text, roi_font)
        # y=200 is the top of the block per spec.
        draw.text(((w - r_w) / 2 - r_ox, 200 - r_oy), roi_text, font=roi_font, fill=pnl_color)

        pnl_text = f"{_fmt_signed(focus.unrealized_pnl, 2)} USDC"
        pnl_font = _font(64, bold=True)
        p_w, p_h, p_ox, p_oy = _text_wh(draw, pnl_text, pnl_font)
        draw.text(((w - p_w) / 2 - p_ox, 450 - p_oy), pnl_text, font=pnl_font, fill=pnl_color)
    else:
        empty_text = "No open positions"
        empty_font = _font(60, bold=False)
        e_w, e_h, e_ox, e_oy = _text_wh(draw, empty_text, empty_font)
        draw.text(
            ((w - e_w) / 2 - e_ox, (h - e_h) / 2 - e_oy),
            empty_text,
            font=empty_font,
            fill=_COL_GRAY,
        )
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    # --- Bottom-left: Entry Price / Mark Price ---------------------------
    label_font = _font(30, bold=False)
    lbl2_font = _font(22, bold=False)
    value_font = _font(56, bold=True)
    ref_code_font = _font(52, bold=True)

    draw.text((60, h - 200), "Entry Price", font=label_font, fill=_COL_GRAY)
    draw.text(
        (60, h - 155),
        _fmt_price(focus.entry_price),
        font=value_font,
        fill=_COL_WHITE,
    )

    draw.text((500, h - 200), "Mark Price", font=label_font, fill=_COL_GRAY)
    draw.text(
        (500, h - 155),
        _fmt_price(focus.mark_price),
        font=value_font,
        fill=_COL_WHITE,
    )

    # --- Bottom-right: REFERRAL block ------------------------------------
    pnl_color = _COL_PROFIT if focus.unrealized_pnl >= 0 else _COL_LOSS
    ref_code = "HhNG2"  # placeholder per spec; real code plumbed later
    ref_url = f"app.propr.xyz/r/{ref_code}"

    ref_lbl_w, _, ref_lbl_ox, ref_lbl_oy = _text_wh(draw, "REFERRAL", lbl2_font)
    code_w, _, code_ox, code_oy = _text_wh(draw, ref_code, ref_code_font)
    url_w, _, url_ox, url_oy = _text_wh(draw, ref_url, _font(26, bold=False))

    right_x = w - 60
    draw.text(
        (right_x - ref_lbl_w - ref_lbl_ox, h - 210 - ref_lbl_oy),
        "REFERRAL",
        font=lbl2_font,
        fill=_COL_GRAY,
    )
    draw.text(
        (right_x - code_w - code_ox, h - 175 - code_oy),
        ref_code,
        font=ref_code_font,
        fill=_COL_WHITE,
    )
    draw.text(
        (right_x - url_w - url_ox, h - 95 - url_oy),
        ref_url,
        font=_font(26, bold=False),
        fill=pnl_color,
    )

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Profit variant \u2014 single long BTC position matching the spec's example:
    # entry 74386.00, mark 75359.50, uPnL +65.44, margin_used 1006.77, lev 5.
    # ROI should format as +6.5%.
    fake_profit = [
        {
            "positionId": "pos-btc",
            "asset": "BTC",
            "positionSide": "long",
            "status": "open",
            "quantity": "0.068",
            "entryPrice": "74386.00",
            "markPrice": "75359.50",
            "unrealizedPnl": "65.44",
            "realizedPnl": "0",
            "leverage": "5",
            "marginUsed": "1006.77",
            "returnOnEquity": "0.065",
        }
    ]

    fake_loss = [
        {
            "positionId": "pos-eth",
            "asset": "ETH",
            "positionSide": "short",
            "status": "open",
            "quantity": "0.25",
            "entryPrice": "3480.50",
            "markPrice": "3594.80",
            "unrealizedPnl": "-28.70",
            "realizedPnl": "0",
            "leverage": "3",
            "marginUsed": "668.76",
            "returnOnEquity": "-0.042",
        }
    ]

    snap_profit = build_snapshot(fake_profit)
    snap_loss = build_snapshot(fake_loss)

    png_profit = render_snapshot_image(snap_profit)
    png_loss = render_snapshot_image(snap_loss)

    out_profit = "/tmp/pnl_share_profit.png"
    out_loss = "/tmp/pnl_share_loss.png"
    with open(out_profit, "wb") as fh:
        fh.write(png_profit)
    with open(out_loss, "wb") as fh:
        fh.write(png_loss)

    print(f"profit: {out_profit} ({len(png_profit)} bytes)")
    print(f"loss:   {out_loss} ({len(png_loss)} bytes)")

    for label, payload in ((out_profit, png_profit), (out_loss, png_loss)):
        if not payload.startswith(b"\x89PNG"):
            print(f"ERROR: {label} is not a PNG", file=sys.stderr)
            sys.exit(1)
        if len(payload) <= 5000:
            print(f"ERROR: {label} too small ({len(payload)} bytes)", file=sys.stderr)
            sys.exit(1)
    print("OK")
