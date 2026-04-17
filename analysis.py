"""Expert AI trading analysis: Groq brain + market data + recommendation parser.

This module is responsible for three things:

1. Gathering all context (account, positions, challenge rules, market data)
   needed to ask Groq for a high-quality analysis.
2. Calling Groq (``llama-3.3-70b-versatile``) with a strict system prompt and
   a templated user message.
3. Parsing the structured response back into a machine-readable dict so the
   Telegram callback handler can auto-execute the trade.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

import httpx
from groq import Groq

from propr import (
    HIP3_ASSETS,
    ProprAPIError,
    ProprClient,
    asset_ticker,
    max_leverage_for,
    normalize_asset,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Market data mapping
# ---------------------------------------------------------------------------
BINANCE_SYMBOLS: Dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "DOGE": "DOGEUSDT",
    "XRP": "XRPUSDT",
    "ADA": "ADAUSDT",
    "AVAX": "AVAXUSDT",
    "LINK": "LINKUSDT",
    "LTC": "LTCUSDT",
    "MATIC": "MATICUSDT",
    "ARB": "ARBUSDT",
    "OP": "OPUSDT",
    "SUI": "SUIUSDT",
    "APT": "APTUSDT",
    "SEI": "SEIUSDT",
    "TIA": "TIAUSDT",
    "INJ": "INJUSDT",
    "PEPE": "PEPEUSDT",
    "WIF": "WIFUSDT",
    "BONK": "BONKUSDT",
}

VALID_INTERVALS = {"15m", "1h", "4h", "1d"}

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a professional prop firm trader and quantitative analyst with deep
expertise in crypto perpetual futures on Hyperliquid. You have 10+ years
of experience trading BTC, ETH, and altcoin perps. You are disciplined,
risk-aware, and always consider the prop firm challenge rules before
recommending any trade.

Your analysis style:
- Think like a professional: structure, risk, edge, invalidation
- Always identify key levels: support, resistance, liquidity zones
- Always define the trade setup: entry zone, stop loss, take profit targets
- Always calculate risk/reward ratio
- Always check if the trade respects challenge rules (drawdown, daily loss)
- Be direct: end every analysis with a clear RECOMMENDATION

Your output must always follow this EXACT structure with these EXACT headers:

\U0001f4ca MARKET STRUCTURE
[Trend, key S/R levels, current price context based on the data provided]

\U0001f4f0 SENTIMENT & CATALYSTS
[Funding rate interpretation, fear/greed reading, market mood]

\U0001f3af TRADE SETUP \u2014 [LONG / SHORT / NO TRADE]
Entry zone: ...
Stop loss: ... (X% risk)
Target 1: ... (R:R X)
Target 2: ... (R:R X)
Target 3: ... (optional)

\u26a0\ufe0f RISK ASSESSMENT
[Challenge rules check: daily loss headroom, drawdown headroom]
[Position sizing recommendation given current account state]
[Key invalidation level]

\U0001f9e0 REASONING
[3-5 bullet points explaining the conviction]

\u2705 RECOMMENDATION: [STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL / WAIT]
Confidence: [HIGH / MEDIUM / LOW]
Suggested size: [e.g. 0.001 BTC]
Suggested leverage: [e.g. 2x]

If this is a NO TRADE call, explain exactly what condition you are waiting for.
Never deviate from this output format. It will be parsed programmatically."""


# ---------------------------------------------------------------------------
# Intent parser — converts free-form chat into a structured trade intent.
# ---------------------------------------------------------------------------
INTENT_SYSTEM_PROMPT = """You are a strict JSON intent parser for a crypto trading bot on Hyperliquid.
Your ONLY output is a single JSON object. No prose, no code fences, no markdown.

Supported intents:
  - "open"          open a new position
  - "close"         fully close an existing position
  - "sl"            attach stop-loss to an existing position
  - "tp"            attach take-profit to an existing position
  - "setleverage"   change leverage on an asset
  - "analysis"      run AI analysis on an asset
  - "unknown"       intent unclear

Schema (fields not applicable must be null):
{
  "intent": "open|close|sl|tp|setleverage|analysis|unknown",
  "asset": "BTC | ETH | SOL | ... | null",
  "side": "long|short|null",
  "order_type": "market|limit|null",
  "quantity": "<decimal as string>|null",
  "usd_amount": "<decimal as string>|null",
  "leverage": "<integer>|null",
  "limit_price": "<decimal as string>|null",
  "stop_loss": "<decimal as string>|null",
  "take_profit": "<decimal as string>|null",
  "timeframe": "15m|1h|4h|1d|null",
  "notes": "<short human-readable paraphrase of the request>"
}

Rules:
- For "open": exactly one of quantity or usd_amount must be set.
- usd_amount is the notional in USDC the user wants to deploy (e.g. "$4000" -> "4000").
- leverage is an integer; default null if unspecified.
- order_type is "limit" when a specific price is given (e.g. "buy btc at 92000"), otherwise "market".
- If the user says "close my btc", intent="close", asset="BTC", side can be null (we'll figure it out).
- If you can't map the text to a supported intent, return intent="unknown" with a polite `notes` explaining why.
- Assets: BTC, ETH, SOL, DOGE, XRP, ADA, AVAX, LINK, LTC, MATIC, ARB, OP, SUI, APT, SEI, TIA, INJ, PEPE, WIF, BONK, AAPL, TSLA, NVDA, MSFT, META, GOOGL, AMZN, GOLD, SILVER, OIL.
- Never invent prices. If user didn't say a price, leave it null.
- Output MUST be valid JSON parsable by json.loads.
"""

VALID_INTENT_KINDS = frozenset({
    "open", "close", "sl", "tp", "setleverage", "analysis", "unknown",
})


def _unknown_intent(reason: str) -> Dict[str, Any]:
    """Return the canonical ``intent=unknown`` sentinel with a reason in notes."""
    return {
        "intent": "unknown",
        "asset": None,
        "side": None,
        "order_type": None,
        "quantity": None,
        "usd_amount": None,
        "leverage": None,
        "limit_price": None,
        "stop_loss": None,
        "take_profit": None,
        "timeframe": None,
        "notes": f"Could not parse: {reason}",
    }


_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?|\n?```\s*$", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    """Remove accidental ```json ... ``` fences the model may emit."""
    if not text:
        return ""
    # Greedy trim of leading/trailing fences; inner backticks are left alone.
    out = text.strip()
    if out.startswith("```"):
        out = _CODE_FENCE_RE.sub("", out, count=1)
    if out.endswith("```"):
        out = _CODE_FENCE_RE.sub("", out, count=1)
    return out.strip()


def _call_groq_intent_sync(api_key: str, text: str) -> str:
    """Synchronous Groq call for intent parsing — invoked via ``asyncio.to_thread``."""
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


async def parse_trade_intent(text: str) -> Dict[str, Any]:
    """Run the user's free-form text through Groq and return the parsed intent.

    Returns the JSON decoded as a dict. On any failure (JSON parse error, Groq
    error, empty response), returns ``{"intent": "unknown", ...}`` with a
    ``notes`` field explaining why — the caller never crashes.

    Decimal-valued fields are left as strings in the returned dict; convert
    at the caller so schema changes don't ripple.
    """
    if not text or not text.strip():
        return _unknown_intent("empty message")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _unknown_intent("GROQ_API_KEY not set")

    try:
        raw = await asyncio.to_thread(_call_groq_intent_sync, api_key, text)
    except Exception as exc:  # noqa: BLE001 — groq/sdk/network surface too many types
        log.warning("parse_trade_intent Groq call failed: %s", exc)
        return _unknown_intent(f"model error: {exc}")

    stripped = _strip_code_fences(raw)
    if not stripped:
        return _unknown_intent("empty model response")

    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError) as exc:
        log.warning("parse_trade_intent JSON decode failed: %s | raw=%r", exc, raw[:500])
        return _unknown_intent("invalid JSON from model")

    if not isinstance(parsed, dict):
        return _unknown_intent("model returned non-object")

    intent = str(parsed.get("intent", "")).lower().strip()
    if intent not in VALID_INTENT_KINDS:
        parsed["intent"] = "unknown"
        parsed.setdefault(
            "notes",
            f"Unsupported intent {intent!r}",
        )
    else:
        parsed["intent"] = intent

    # Normalize asset to upper-case; leave Decimal-looking fields as strings.
    asset = parsed.get("asset")
    if isinstance(asset, str) and asset:
        parsed["asset"] = asset.split(":")[-1].upper()

    side = parsed.get("side")
    if isinstance(side, str):
        lowered = side.lower().strip()
        parsed["side"] = lowered if lowered in ("long", "short") else None

    order_type = parsed.get("order_type")
    if isinstance(order_type, str):
        lowered = order_type.lower().strip()
        parsed["order_type"] = lowered if lowered in ("market", "limit") else None

    timeframe = parsed.get("timeframe")
    if isinstance(timeframe, str):
        lowered = timeframe.lower().strip()
        parsed["timeframe"] = lowered if lowered in VALID_INTERVALS else None

    return parsed


async def fetch_spot_price(asset: str) -> Optional[Decimal]:
    """Fetch the latest spot price (USDT quote) from Binance.

    Returns ``None`` on any network/parse failure so the caller can degrade
    gracefully. Never raises.

    HIP-3 assets (equities/commodities like AAPL, GOLD, xyz:CL) don't trade
    on Binance under a ``{TICKER}USDT`` pair, so we short-circuit with a
    single info log instead of hammering the Binance endpoint with
    guaranteed-404 requests (review finding 8).
    """
    if not asset:
        return None
    # review finding 8: skip Binance for HIP-3 tickers — no USDT pair exists.
    ticker = asset_ticker(asset)
    if asset.startswith("xyz:") or ticker in HIP3_ASSETS:
        log.info("fetch_spot_price skipped for HIP-3 asset %s", asset)
        return None
    symbol = binance_symbol(asset)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": symbol},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001 — any failure degrades to None
        log.warning("fetch_spot_price %s failed: %s", symbol, exc)
        return None
    price = data.get("price") if isinstance(data, dict) else None
    if price is None:
        log.warning("fetch_spot_price %s missing 'price': %s", symbol, data)
        return None
    try:
        return Decimal(str(price))
    except (InvalidOperation, ValueError, TypeError) as exc:
        log.warning("fetch_spot_price %s decimal parse failed: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Pending trade store (module-level, swept every 30s from bot.py)
# ---------------------------------------------------------------------------
@dataclass
class PendingTrade:
    """A parsed Groq analysis awaiting user confirmation via inline buttons."""

    pending_id: str
    chat_id: int
    asset: str
    timeframe: str
    raw_analysis: str
    parsed: Dict[str, Any]
    created_at: datetime
    message_id: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


PENDING_TRADES: Dict[str, PendingTrade] = {}
PENDING_TTL_SECONDS = 300  # 5 minutes


@dataclass
class PendingIntent:
    """A parsed NL trade intent awaiting Execute/Cancel confirmation."""

    pending_id: str
    chat_id: int
    intent: Dict[str, Any]              # the parsed intent from Groq
    resolved_qty: Optional[Decimal]     # quantity after usd_amount → qty conversion
    price_at_parse: Optional[Decimal]   # spot price used to compute resolved_qty
    created_at: datetime                # UTC
    message_id: Optional[int] = None
    notes: List[str] = field(default_factory=list)  # user-visible quirks (clamps, defaults)
    # review finding 14: keep the original MarkdownV2 card body so the
    # sweeper can append ``⏰ Confirmation expired`` to it on TTL.
    message_text: Optional[str] = None


PENDING_INTENTS: Dict[str, PendingIntent] = {}


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def binance_symbol(asset: str) -> str:
    """Map an asset symbol to the Binance ticker used for market data."""
    base = asset.split(":")[-1].upper()
    return BINANCE_SYMBOLS.get(base, f"{base}USDT")


async def fetch_market_data(asset: str, timeframe: str = "1h") -> Dict[str, Any]:
    """Fetch ticker, klines, funding rate and fear-and-greed in parallel.

    On partial failure we fall back to ``"unavailable"`` values for the
    affected fields so the Groq prompt can still be constructed.
    """
    symbol = binance_symbol(asset)
    interval = timeframe if timeframe in VALID_INTERVALS else "1h"

    async with httpx.AsyncClient(timeout=10.0) as client:
        results = await asyncio.gather(
            _get_json(client, "https://api.binance.com/api/v3/ticker/24hr",
                      {"symbol": symbol}),
            _get_json(client, "https://api.binance.com/api/v3/klines",
                      {"symbol": symbol, "interval": interval, "limit": 50}),
            _get_json(client, "https://fapi.binance.com/fapi/v1/fundingRate",
                      {"symbol": symbol, "limit": 5}),
            _get_json(client, "https://api.alternative.me/fng/",
                      {"limit": 3}),
            return_exceptions=True,
        )

    ticker, klines, funding, fng = results

    data: Dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "current_price": "unavailable",
        "change_24h_pct": "unavailable",
        "recent_high": "unavailable",
        "recent_low": "unavailable",
        "last_5_closes": [],
        "funding_rate": "unavailable",
        "funding_interpretation": "unavailable",
        "fg_value": "unavailable",
        "fg_classification": "unavailable",
    }

    # 24h ticker
    if isinstance(ticker, dict) and ticker.get("lastPrice"):
        try:
            data["current_price"] = float(ticker["lastPrice"])
            data["change_24h_pct"] = float(ticker.get("priceChangePercent", 0))
        except (TypeError, ValueError):
            pass

    # Klines: [open_time, open, high, low, close, volume, ...]
    if isinstance(klines, list) and klines:
        try:
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            closes = [float(k[4]) for k in klines]
            data["recent_high"] = max(highs)
            data["recent_low"] = min(lows)
            data["last_5_closes"] = [round(c, 4) for c in closes[-5:]]
        except (TypeError, ValueError, IndexError) as exc:
            log.debug("klines parse failed: %s", exc)

    # Funding rate
    if isinstance(funding, list) and funding:
        try:
            rate = float(funding[-1]["fundingRate"])
            pct = rate * 100
            data["funding_rate"] = round(pct, 5)
            if pct > 0.002:
                data["funding_interpretation"] = "longs pay shorts"
            elif pct < -0.002:
                data["funding_interpretation"] = "shorts pay longs"
            else:
                data["funding_interpretation"] = "neutral"
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            log.debug("funding parse failed: %s", exc)

    # Fear & Greed
    if isinstance(fng, dict):
        rows = fng.get("data") or []
        if rows:
            try:
                data["fg_value"] = int(rows[0]["value"])
                data["fg_classification"] = rows[0].get(
                    "value_classification", "unavailable"
                )
            except (TypeError, ValueError, KeyError):
                pass

    return data


async def _get_json(
    client: httpx.AsyncClient, url: str, params: Dict[str, Any]
) -> Any:
    """GET *url* and return parsed JSON, or an Exception on failure."""
    try:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 — downstream code tolerates any failure
        log.warning("market fetch %s failed: %s", url, exc)
        return exc


# ---------------------------------------------------------------------------
# Account context
# ---------------------------------------------------------------------------
async def gather_account_context(
    client: ProprClient, account_id: str, asset: str
) -> Dict[str, Any]:
    """Collect all account-side inputs needed by the Groq prompt."""
    asset_upper = asset.split(":")[-1].upper()
    asset_norm = normalize_asset(asset)

    positions, orders, trades, attempts, me = await asyncio.gather(
        client.get_positions(account_id),
        client.get_open_orders(account_id),
        client.get_trades(account_id, limit=10),
        client.get_challenge_attempts(),
        client.me(),
        return_exceptions=True,
    )

    def _ok(v: Any) -> Any:
        if isinstance(v, Exception):
            log.warning("context fetch failed: %s", v)
            return None
        return v

    positions = _ok(positions) or []
    orders = _ok(orders) or []
    trades = _ok(trades) or []
    attempts = _ok(attempts) or []
    me = _ok(me) or {}

    # Win rate on this asset specifically (fall back to all trades if none).
    asset_trades = [
        t
        for t in trades
        if str(t.get("asset", "")).split(":")[-1].upper() == asset_upper
    ]
    pool = asset_trades or trades
    wins = sum(1 for t in pool if _as_decimal(t.get("pnl", 0)) > 0)
    win_rate = (wins / len(pool) * 100) if pool else 0.0

    # Challenge fields — some missing on Propr's side get treated as 0.
    current_attempt: Dict[str, Any] = {}
    if attempts:
        def _active(a: Dict[str, Any]) -> bool:
            s = str(a.get("status", "")).lower()
            return s in ("", "active", "in_progress", "running", "open")
        current_attempt = next((a for a in attempts if _active(a)), attempts[0])

    drawdown_used = _as_decimal(current_attempt.get("drawdownUsed", 0))
    max_drawdown = _as_decimal(current_attempt.get("maxDrawdown", 0))
    daily_loss_used = _as_decimal(current_attempt.get("dailyLossUsed", 0))
    max_daily_loss = _as_decimal(current_attempt.get("maxDailyLoss", 0))
    profit = _as_decimal(current_attempt.get("profit", 0))
    profit_target = _as_decimal(current_attempt.get("profitTarget", 0))

    drawdown_pct = _safe_pct(drawdown_used, max_drawdown)
    daily_loss_pct = _safe_pct(daily_loss_used, max_daily_loss)
    profit_pct = _safe_pct(profit, profit_target)

    # Balance + margin used
    balance = _as_decimal(
        current_attempt.get("balance")
        or current_attempt.get("equity")
        or me.get("balance", 0)
    )
    margin_used = sum(
        (_as_decimal(p.get("marginUsed", p.get("margin", 0))) for p in positions),
        Decimal(0),
    )

    # Orders filtered to this asset
    asset_orders = [
        o
        for o in orders
        if str(o.get("asset", "")).split(":")[-1].upper() == asset_upper
    ]

    # Leverage cap
    max_lev = max_leverage_for(asset_norm)
    try:
        lev_limits = await client.get_leverage_limits()
        if isinstance(lev_limits, dict):
            api_cap = lev_limits.get(asset_upper) or lev_limits.get(asset_norm)
            if api_cap:
                try:
                    api_cap_int = int(float(api_cap))
                    max_lev = min(max_lev, api_cap_int)
                except (TypeError, ValueError):
                    pass
    except Exception as exc:  # noqa: BLE001
        log.debug("leverage limits fetch failed: %s", exc)

    return {
        "positions": positions,
        "orders": asset_orders,
        "trades": trades,
        "win_rate": win_rate,
        "balance": balance,
        "margin_used": margin_used,
        "drawdown_pct": drawdown_pct,
        "daily_loss_pct": daily_loss_pct,
        "profit_pct": profit_pct,
        "max_leverage": max_lev,
    }


def _as_decimal(value: Any) -> Decimal:
    """Coerce *value* to :class:`Decimal`; return 0 on failure."""
    if value is None:
        return Decimal(0)
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(0)


def _safe_pct(numerator: Decimal, denominator: Decimal) -> float:
    """Return ``numerator / denominator * 100`` or ``0.0`` if denominator is 0."""
    if denominator == 0:
        return 0.0
    try:
        return float(numerator / denominator * 100)
    except (InvalidOperation, ZeroDivisionError):
        return 0.0


# ---------------------------------------------------------------------------
# Groq call
# ---------------------------------------------------------------------------
def _format_positions(positions: List[Dict[str, Any]]) -> str:
    """Render the positions list as a one-line summary for the prompt."""
    if not positions:
        return ""
    parts = []
    for p in positions:
        asset = p.get("asset", "?")
        side = p.get("positionSide") or p.get("side") or "?"
        qty = p.get("quantity", p.get("qty", "?"))
        entry = p.get("entryPrice", p.get("avgEntryPrice", "?"))
        upnl = p.get("unrealizedPnl", p.get("uPnl", "?"))
        parts.append(f"{side} {qty} {asset} @ {entry} (uPnL {upnl})")
    return "; ".join(parts)


def _format_orders(orders: List[Dict[str, Any]]) -> str:
    """Render the open orders list as a one-line summary for the prompt."""
    if not orders:
        return ""
    parts = []
    for o in orders:
        parts.append(
            f"{o.get('type', '?')} {o.get('side', '?')} "
            f"{o.get('quantity', '?')} @ {o.get('price', '?')}"
        )
    return "; ".join(parts)


def build_user_message(
    asset: str,
    timeframe: str,
    direction: Optional[str],
    market: Dict[str, Any],
    context: Dict[str, Any],
) -> str:
    """Render the Groq user message from market data + account context."""
    positions_summary = _format_positions(context.get("positions", []))
    orders_summary = _format_orders(context.get("orders", []))

    change_pct = market.get("change_24h_pct", "unavailable")
    change_str = (
        f"{change_pct:+.2f}% 24h" if isinstance(change_pct, (int, float)) else "unavailable"
    )

    return (
        f"Asset: {asset}\n"
        f"Timeframe: {timeframe}\n"
        f"User direction hint: {direction or 'none — analyze independently'}\n\n"
        f"MARKET DATA (fetched live):\n"
        f"Current price: {market.get('current_price')} ({change_str})\n"
        f"Recent high (50 candles): {market.get('recent_high')}\n"
        f"Recent low (50 candles): {market.get('recent_low')}\n"
        f"Last 5 closes: {market.get('last_5_closes')}\n"
        f"Funding rate: {market.get('funding_rate')}% "
        f"({market.get('funding_interpretation')})\n"
        f"Fear & Greed: {market.get('fg_value')} — {market.get('fg_classification')}\n\n"
        f"ACCOUNT STATE:\n"
        f"Balance: {context.get('balance')} USDC\n"
        f"Margin used: {context.get('margin_used')} USDC\n"
        f"Challenge drawdown used: {context.get('drawdown_pct', 0):.1f}% of max\n"
        f"Challenge daily loss used: {context.get('daily_loss_pct', 0):.1f}% of max\n"
        f"Challenge profit target progress: {context.get('profit_pct', 0):.1f}%\n"
        f"Open positions: {positions_summary or 'none'}\n"
        f"Open orders on {asset}: {orders_summary or 'none'}\n"
        f"Last 10 trades win rate: {context.get('win_rate', 0):.0f}%\n\n"
        f"Max leverage for {asset}: {context.get('max_leverage')}x\n\n"
        f"Perform your full expert analysis and give your recommendation."
    )


def _call_groq_sync(api_key: str, user_message: str) -> str:
    """Synchronous Groq call — invoked via ``asyncio.to_thread``."""
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=1500,
    )
    return response.choices[0].message.content or ""


async def run_analysis(
    propr_client: ProprClient,
    account_id: str,
    asset: str,
    timeframe: str = "1h",
    direction: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full analysis pipeline: context + market → Groq → parsed dict.

    Returns a dict with ``asset``, ``timeframe``, ``raw`` (full Groq text),
    ``parsed`` (output of :func:`parse_groq_recommendation`) and
    ``market_fetched_at`` (ISO UTC timestamp string).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    market_task = asyncio.create_task(fetch_market_data(asset, timeframe))
    context_task = asyncio.create_task(
        gather_account_context(propr_client, account_id, asset)
    )
    market, context = await asyncio.gather(market_task, context_task)

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    user_message = build_user_message(asset, timeframe, direction, market, context)

    log.debug("Calling Groq for %s (%s)", asset, timeframe)
    raw = await asyncio.to_thread(_call_groq_sync, api_key, user_message)

    parsed = parse_groq_recommendation(raw)
    return {
        "asset": asset,
        "timeframe": timeframe,
        "raw": raw,
        "parsed": parsed,
        "market_fetched_at": fetched_at,
        "market": market,
        "context": context,
    }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
NUMBER_RE = re.compile(r"(\d+(?:[,_]\d{3})*(?:\.\d+)?)")


def _first_decimal(text: str) -> Optional[Decimal]:
    """Return the first number in *text* as a :class:`Decimal`."""
    m = NUMBER_RE.search(text)
    if not m:
        return None
    raw = m.group(1).replace(",", "").replace("_", "")
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _all_decimals(text: str) -> List[Decimal]:
    """Return every number in *text* as a list of :class:`Decimal`."""
    out: List[Decimal] = []
    for m in NUMBER_RE.finditer(text):
        raw = m.group(1).replace(",", "").replace("_", "")
        try:
            out.append(Decimal(raw))
        except InvalidOperation:
            continue
    return out


def _line_after(text: str, label: str) -> Optional[str]:
    """Return the trimmed remainder of the line starting with *label*.

    Match is case-insensitive and tolerates leading whitespace, bullets,
    asterisks and any non-alphanumeric prefix (emoji included).
    """
    pattern = re.compile(
        rf"^[^\w\n]*{re.escape(label)}\s*[:\-–—]?\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _parse_direction(text: str) -> str:
    """Extract ``long`` / ``short`` / ``no_trade`` from the TRADE SETUP line."""
    m = re.search(
        r"TRADE\s*SETUP\s*[\-\u2013\u2014:]*\s*[\[\(]?\s*(LONG|SHORT|NO\s*TRADE)",
        text,
        re.IGNORECASE,
    )
    if m:
        tag = re.sub(r"\s+", "_", m.group(1).strip().lower())
        if tag in ("long", "short"):
            return tag
        return "no_trade"

    # Fall back to the RECOMMENDATION line
    rec = _line_after(text, "RECOMMENDATION")
    if rec:
        rec_upper = rec.upper()
        if "WAIT" in rec_upper or "NEUTRAL" in rec_upper:
            return "no_trade"
        if "STRONG BUY" in rec_upper or rec_upper.startswith("BUY"):
            return "long"
        if "STRONG SELL" in rec_upper or rec_upper.startswith("SELL"):
            return "short"
    return "no_trade"


# Accept only these exact tokens as "market" entry; anything else with a
# digit is treated as a limit price. See PR #1 review finding #4.
_MARKET_ENTRY_TOKENS = frozenset({"market", "market order", "at market"})


def _parse_entry(text: str) -> tuple[Optional[str], Optional[Decimal]]:
    """Return ``(order_type, entry_price)`` parsed from ``Entry zone:``.

    - Exact ``market`` / ``market order`` / ``at market`` (with no digits on
      the line) → ``("market", None)``.
    - Single number → ``("limit", Decimal)``.
    - Range of two numbers → ``("limit", midpoint)``.
    - No digits and not an exact market token (e.g. ``"wait for retest"``)
      → ``(None, None)`` so the caller marks the trade non-executable
      (review finding 6).

    Prior behavior treated any substring "market" as market (PR #1 finding
    #4) — so ``"sweep the market at 94500"`` turned into a market order.
    Further tightened by review finding 6: free-form waiting text no
    longer silently maps to a market order.
    """
    # review finding 6: require an exact market token; otherwise demand
    # a digit on the line or return (None, None) as non-executable.
    line = _line_after(text, "Entry zone")
    if not line:
        line = _line_after(text, "Entry")
    if not line:
        return None, None

    stripped = line.strip().lower()
    if stripped in _MARKET_ENTRY_TOKENS and not any(c.isdigit() for c in line):
        return "market", None

    nums = _all_decimals(line)
    if not nums:
        return None, None
    low = stripped
    if len(nums) >= 2 and ("-" in line or "–" in line or "—" in line or "to" in low):
        mid = (nums[0] + nums[1]) / Decimal(2)
        return "limit", mid
    return "limit", nums[0]


# Tokens that signal the suggested size is a percentage / notional and NOT a
# raw quantity we can pass to the exchange; reject these to avoid placing
# massively wrong orders. See PR #1 review finding #5, plus review
# finding 11 which added plain-English magnitude words.
_SIZE_NON_QUANTITY_TOKENS = (
    "%", "percent", "$", "usd", "usdc", "notional",
    "of account", "of balance",
    # review finding 11: reject English-word notionals/magnitudes.
    "worth", "equivalent", "hundred", "thousand", "million",
)
# review finding 7: accept thousands-separated integer parts
# (``"1,000 DOGE"``) by allowing ``1-3 digits`` + ``,{3 digits}``.
_SIZE_QUANTITY_RE = re.compile(
    r"^\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*([A-Z]{2,10})?\s*$"
)


def _parse_size(text: str) -> Optional[Decimal]:
    """Extract quantity (first number) from ``Suggested size:``.

    Returns ``None`` when the line looks like a percentage / notional / USDC
    value (the caller treats ``None`` as non-executable). Also returns
    ``None`` if the line doesn't match ``<number> <asset>?`` — anything
    shaped unambiguously like a raw quantity.
    """
    line = _line_after(text, "Suggested size")
    if not line:
        return None
    lowered = line.lower()
    if any(tok in lowered for tok in _SIZE_NON_QUANTITY_TOKENS):
        return None
    if not _SIZE_QUANTITY_RE.match(line):
        return None
    # review finding 7: strip thousands commas before Decimal coercion.
    cleaned = line.replace(",", "")
    return _first_decimal(cleaned)


def _parse_leverage(text: str) -> Optional[int]:
    """Extract the integer before ``x`` in ``Suggested leverage:``."""
    line = _line_after(text, "Suggested leverage")
    if not line:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*[xX]", line)
    if not m:
        d = _first_decimal(line)
        return int(d) if d is not None else None
    try:
        return int(float(m.group(1)))
    except ValueError:
        return None


def _parse_confidence(text: str) -> Optional[str]:
    """Extract ``HIGH``/``MEDIUM``/``LOW`` from the ``Confidence:`` line."""
    line = _line_after(text, "Confidence")
    if not line:
        return None
    upper = line.upper()
    for level in ("HIGH", "MEDIUM", "LOW"):
        if level in upper:
            return level
    return None


def _parse_recommendation_label(text: str) -> Optional[str]:
    """Extract the recommendation label (e.g. ``STRONG BUY``, ``WAIT``)."""
    line = _line_after(text, "RECOMMENDATION")
    if not line:
        return None
    upper = line.upper()
    for label in ("STRONG BUY", "STRONG SELL", "BUY", "SELL", "NEUTRAL", "WAIT"):
        if label in upper:
            return label
    return None


def parse_groq_recommendation(text: str) -> Dict[str, Any]:
    """Parse a Groq analysis response into a structured dict.

    See the module docstring and the bot spec for the exact contract. If any
    critical execution field is missing (``direction`` not long/short,
    ``quantity`` missing, ``stop_loss`` missing, ``take_profit_1`` missing)
    the returned dict sets ``executable=False``.
    """
    if not text:
        return {
            "direction": "no_trade",
            "order_type": "market",
            "entry_price": None,
            "stop_loss": None,
            "take_profit_1": None,
            "take_profit_2": None,
            "quantity": None,
            "leverage": None,
            "confidence": None,
            "recommendation_label": None,
            "executable": False,
        }

    direction = _parse_direction(text)
    order_type, entry_price = _parse_entry(text)

    sl_line = _line_after(text, "Stop loss") or _line_after(text, "Stop-loss")
    stop_loss = _first_decimal(sl_line) if sl_line else None

    tp1_line = _line_after(text, "Target 1")
    take_profit_1 = _first_decimal(tp1_line) if tp1_line else None

    tp2_line = _line_after(text, "Target 2")
    take_profit_2 = _first_decimal(tp2_line) if tp2_line else None

    quantity = _parse_size(text)
    leverage = _parse_leverage(text)
    confidence = _parse_confidence(text)
    recommendation_label = _parse_recommendation_label(text)

    # review finding 6: treat a non-parseable entry (order_type is None) as
    # non-executable so free-form "wait for retest" entries can't auto-fire.
    executable = (
        direction in ("long", "short")
        and order_type is not None
        and quantity is not None
        and stop_loss is not None
        and take_profit_1 is not None
    )

    return {
        "direction": direction,
        "order_type": order_type,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "quantity": quantity,
        "leverage": leverage,
        "confidence": confidence,
        "recommendation_label": recommendation_label,
        "executable": executable,
    }


# ---------------------------------------------------------------------------
# Pending trade sweeper
# ---------------------------------------------------------------------------
async def sweep_pending_trades(bot: Any) -> None:
    """Remove expired pending trades and edit the Telegram message.

    Runs forever — start it as a background task from ``bot.py``. Sweeps both
    :data:`PENDING_TRADES` (``/analysis`` flow) and :data:`PENDING_INTENTS`
    (natural-language ``/trade`` flow) on the same 30s cadence.
    """
    while True:
        try:
            await asyncio.sleep(30)
            now = datetime.now(timezone.utc)

            expired_trades = [
                pid
                for pid, pt in PENDING_TRADES.items()
                if (now - pt.created_at).total_seconds() > PENDING_TTL_SECONDS
            ]
            for pid in expired_trades:
                pt = PENDING_TRADES.pop(pid, None)
                if pt is None:
                    continue
                if pt.message_id is not None:
                    try:
                        await bot.edit_message_reply_markup(
                            chat_id=pt.chat_id,
                            message_id=pt.message_id,
                            reply_markup=None,
                        )
                    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                        log.debug("edit_message_reply_markup failed: %s", exc)
                try:
                    await bot.send_message(
                        chat_id=pt.chat_id,
                        text="⏰ Confirmation expired. Run /analysis again.",
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug("send expired notice failed: %s", exc)

            expired_intents = [
                pid
                for pid, pi in PENDING_INTENTS.items()
                if (now - pi.created_at).total_seconds() > PENDING_TTL_SECONDS
            ]
            for pid in expired_intents:
                pi = PENDING_INTENTS.pop(pid, None)
                if pi is None:
                    continue
                # review finding 14: edit the confirmation card itself so the
                # expired state is visible inline, not just as a follow-up.
                if pi.message_id is not None:
                    if pi.message_text:
                        try:
                            await bot.edit_message_text(
                                chat_id=pi.chat_id,
                                message_id=pi.message_id,
                                text=pi.message_text + "\n\n⏰ Confirmation expired",
                                parse_mode="MarkdownV2",
                                reply_markup=None,
                            )
                        except Exception as exc:  # noqa: BLE001 — failed edit must not abort sweep
                            log.debug("edit expired intent card failed: %s", exc)
                    else:
                        try:
                            await bot.edit_message_reply_markup(
                                chat_id=pi.chat_id,
                                message_id=pi.message_id,
                                reply_markup=None,
                            )
                        except Exception as exc:  # noqa: BLE001
                            log.debug("edit_message_reply_markup failed: %s", exc)
                try:
                    await bot.send_message(
                        chat_id=pi.chat_id,
                        text="⏰ Intent confirmation expired. Send it again.",
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug("send expired intent notice failed: %s", exc)
        except asyncio.CancelledError:
            log.info("pending trade sweeper cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("sweep_pending_trades iteration failed: %s", exc)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _SAMPLE_LONG = """📊 MARKET STRUCTURE
BTC trending up on 1h, breaking above 94,500 resistance. Strong bullish structure.
Key support: 93,200. Key resistance: 96,000.

📰 SENTIMENT & CATALYSTS
Funding rate 0.012% positive (longs pay shorts) — mild bullish crowding.
Fear & Greed: 62 — Greed. Sentiment constructive but not euphoric.

🎯 TRADE SETUP — LONG
Entry zone: 94,500-94,800 (scale in)
Stop loss: 93,100 (1.8% risk)
Target 1: 96,200 (R:R 1.5)
Target 2: 97,500 (R:R 2.3)
Target 3: 98,800 (R:R 3.1)

⚠️ RISK ASSESSMENT
Daily loss used 1.2%, drawdown used 3.4% — plenty of headroom.
Suggested risk per trade: 0.5% of balance. Invalidation: close below 93,100.

🧠 REASONING
- Higher highs and higher lows on 1h
- Funding not over-extended
- Fear & Greed in constructive zone
- Win rate strong on BTC longs recently
- Clean invalidation below swing low

✅ RECOMMENDATION: BUY
Confidence: HIGH
Suggested size: 0.002 BTC
Suggested leverage: 3x
"""

    _SAMPLE_NO_TRADE = """📊 MARKET STRUCTURE
ETH consolidating between 3,450 and 3,520. No clear trend on 1h.

📰 SENTIMENT & CATALYSTS
Funding neutral. Fear & Greed: 50 — Neutral.

🎯 TRADE SETUP — NO TRADE
Entry zone: wait for breakout
Stop loss: n/a
Target 1: n/a

⚠️ RISK ASSESSMENT
No edge. Preserve capital.

🧠 REASONING
- Range-bound price action
- Low volatility
- No catalyst

✅ RECOMMENDATION: WAIT
Confidence: MEDIUM
Suggested size: n/a
Suggested leverage: n/a

Waiting for a clean break of 3,520 with volume, or rejection at 3,450.
"""

    def _dump(label: str, result: Dict[str, Any]) -> None:
        print(f"--- {label} ---")
        # Decimal → str so json.dumps works
        safe = {k: (str(v) if isinstance(v, Decimal) else v) for k, v in result.items()}
        print(json.dumps(safe, indent=2))

    _dump("LONG sample", parse_groq_recommendation(_SAMPLE_LONG))
    _dump("NO TRADE sample", parse_groq_recommendation(_SAMPLE_NO_TRADE))

    # Intent parser smoke test — runs only if GROQ_API_KEY is set so the
    # module stays importable/runnable offline.
    _INTENT_SAMPLES = [
        "open long on btc with $4000 and 5x leverage",
        "close my eth",
        "analyze sol 4h",
    ]

    if not os.getenv("GROQ_API_KEY"):
        print("--- intent parser smoke: skipped (GROQ_API_KEY not set) ---")
    else:
        print("--- intent parser smoke ---")
        logging.basicConfig(level=logging.INFO)

        async def _run_intent_smoke() -> None:
            for sample in _INTENT_SAMPLES:
                result = await parse_trade_intent(sample)
                print(f"input: {sample!r}")
                print(json.dumps(result, indent=2, default=str))

        asyncio.run(_run_intent_smoke())
