"""Propr websocket listener — pushes fill/position events to Telegram.

Connects to ``wss://api.propr.xyz/ws`` with the ``X-API-Key`` header, parses
each incoming JSON frame, maps known events onto friendly Telegram messages,
and auto-reconnects on any connection error. A per-message try/except keeps
a handler bug from killing the loop.

On top of the fill/position lifecycle events this listener optionally
streams real-time PnL updates: when ``app.bot_data['live_pnl_enabled']``
is ``True`` and a ``position.updated`` frame arrives, the unrealized PnL
change is pushed to the authorized chat subject to a per-position throttle
(see :func:`_should_push_pnl`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Tuple

import websockets
from telegram.constants import ParseMode
from websockets.exceptions import (
    ConnectionClosed,
    InvalidHandshake,
    InvalidStatusCode,
    WebSocketException,
)

from utils import escape_md, fmt_num

log = logging.getLogger(__name__)

WS_URL = "wss://api.propr.xyz/ws"
RECONNECT_DELAY = 5  # seconds

# Throttle knobs for the live-PnL push.
_LIVE_PNL_MIN_INTERVAL_S = 60        # always push at least every minute
_LIVE_PNL_MIN_DIFF_INTERVAL_S = 10   # "material move" minimum interval
_LIVE_PNL_DIFF_FRACTION = Decimal("0.01")  # 1% of notional

# Last push per positionId: (timestamp_utc, last_unrealized_pnl).
_last_pushed_pnl: Dict[str, Tuple[datetime, Decimal]] = {}

# Per-order fill-event waiters. Populated by :func:`wait_for_fill` and
# resolved inside :func:`_handle_message` when a matching event arrives.
# Keyed by the server-assigned ``orderId`` of the entry order; the future's
# result is the raw event payload dict.
_FILL_WAITERS: Dict[str, "asyncio.Future[Dict[str, Any]]"] = {}
_WAITER_LOCK = asyncio.Lock()

# Events that indicate an entry order is now live enough for conditional
# legs to attach. ``order.triggered`` is included because some venues emit
# it before the canonical ``order.filled`` on limit fills.
_FILL_RESOLVING_EVENTS = frozenset({
    "order.filled",
    "order.partially_filled",
    "order.triggered",
    "position.opened",
})

# Event-name → emoji + template. Keys are lower-cased, punctuation-normalized.
_EVENT_TEMPLATES: Dict[str, str] = {
    "order.filled": "✅ Filled: {side} {qty} {asset} @ {price}",
    "order.partially_filled": "⏳ Partial fill: {filled_qty}/{total_qty} {asset}",
    "order.cancelled": "❌ Order cancelled",
    "position.opened": "📈 Opened: {side} {qty} {asset} @ {entry}",
    "position.closed": "📊 Closed | PnL: {pnl} USDC",
    "position.liquidated": "🚨 LIQUIDATED: {asset} | Loss: {pnl} USDC",
    "position.take_profit.hit": "🎯 TP hit | PnL: {pnl} USDC",
    "position.stop_loss.hit": "🛑 SL hit | PnL: {pnl} USDC",
}


def _canonical_event(name: str) -> str:
    """Normalize event names so camelCase/dotted/snake variants all match.

    ``Position.TakeProfit.Hit`` → ``position.take_profit.hit``.
    """
    if not name:
        return ""
    lowered = []
    for ch in name:
        if ch.isupper():
            if lowered and lowered[-1] not in (".", "_", ""):
                lowered.append("_")
            lowered.append(ch.lower())
        else:
            lowered.append(ch)
    return "".join(lowered)


def _extract_payload(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the event payload — some APIs wrap it in ``data`` or ``payload``."""
    for key in ("data", "payload", "event_data", "body"):
        value = msg.get(key)
        if isinstance(value, dict):
            return {**msg, **value}
    return msg


def _format_event(event: str, payload: Dict[str, Any]) -> Optional[str]:
    """Render *event* using :data:`_EVENT_TEMPLATES`; return ``None`` if unknown."""
    template = _EVENT_TEMPLATES.get(event)
    if template is None:
        return None

    # Provide sensible defaults for missing fields so template formatting
    # cannot crash.
    values = {
        "side": payload.get("side") or payload.get("positionSide") or "?",
        "qty": payload.get("quantity") or payload.get("qty") or "?",
        "filled_qty": payload.get("filledQty")
        or payload.get("filled_quantity")
        or payload.get("filled")
        or "?",
        "total_qty": payload.get("totalQty")
        or payload.get("total_quantity")
        or payload.get("quantity")
        or "?",
        "asset": payload.get("asset") or payload.get("symbol") or "?",
        "price": payload.get("price")
        or payload.get("avgPrice")
        or payload.get("fillPrice")
        or "?",
        "entry": payload.get("entryPrice")
        or payload.get("avgEntryPrice")
        or payload.get("price")
        or "?",
        "pnl": payload.get("pnl")
        or payload.get("realizedPnl")
        or payload.get("unrealizedPnl")
        or "?",
    }
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        log.debug("format error for %s: %s", event, exc)
        return None


# ---------------------------------------------------------------------------
# Live PnL push helpers
# ---------------------------------------------------------------------------
def _dec(value: Any, default: str = "0") -> Decimal:
    """Coerce *value* to :class:`Decimal`; fall back to ``default`` on failure."""
    if value is None or value == "":
        return Decimal(default)
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(default)


def _asset_symbol(raw: str) -> str:
    """Strip any ``xyz:`` venue prefix and upper-case the symbol."""
    if not raw:
        return "?"
    return str(raw).split(":")[-1].upper() or "?"


def _should_push_pnl(
    position_id: str,
    new_pnl: Decimal,
    notional: Decimal,
    now: datetime,
) -> bool:
    """Decide whether to push an updated PnL for *position_id*.

    - Always push if there's no prior entry for this position (first update).
    - Push if ≥ :data:`_LIVE_PNL_MIN_INTERVAL_S` seconds have passed since the
      last push.
    - Push if the absolute PnL change is ≥ 1% of the notional *and* at least
      :data:`_LIVE_PNL_MIN_DIFF_INTERVAL_S` seconds have passed.
    """
    entry = _last_pushed_pnl.get(position_id)
    if entry is None:
        return True
    last_ts, last_pnl = entry
    elapsed = (now - last_ts).total_seconds()
    if elapsed >= _LIVE_PNL_MIN_INTERVAL_S:
        return True
    if elapsed < _LIVE_PNL_MIN_DIFF_INTERVAL_S:
        return False
    diff = abs(new_pnl - last_pnl)
    threshold = abs(notional) * _LIVE_PNL_DIFF_FRACTION
    return threshold > 0 and diff >= threshold


def _signed(value: Decimal, places: int = 2) -> str:
    """Render *value* with a leading ``+``/``-``; bare ``0`` for zero."""
    text = fmt_num(value, places)
    if text == "0" or text.startswith("-"):
        return text
    return "+" + text


def _format_live_pnl(
    asset: str,
    side: str,
    quantity: Decimal,
    entry_price: Decimal,
    mark_price: Decimal,
    unrealized_pnl: Decimal,
) -> str:
    """Build the compact MarkdownV2 live-PnL message."""
    side_upper = side.upper() if side else "?"
    dot = "🟢" if unrealized_pnl >= 0 else "🔴"

    qty_s = fmt_num(quantity, 6)
    entry_s = fmt_num(entry_price)
    mark_s = fmt_num(mark_price)
    upnl_s = _signed(unrealized_pnl)

    notional = quantity * entry_price
    if notional != 0:
        pct = (unrealized_pnl / notional) * Decimal(100)
    else:
        pct = Decimal(0)
    pct_s = _signed(pct)

    return (
        f"{dot} *{escape_md(asset)} {escape_md(side_upper)}* · "
        f"`{qty_s} @ {entry_s}`\n"
        f"Mark: `{mark_s}`  →  uPnL: *{escape_md(upnl_s)} USDC* "
        f"\\({escape_md(pct_s)}%\\)"
    )


async def _maybe_push_live_pnl(
    bot: Any,
    chat_id: int,
    payload: Dict[str, Any],
) -> None:
    """Push a live-PnL message for this ``position.updated`` payload if due.

    Bad / partial payloads are silently ignored (logged at debug) so one
    malformed frame cannot break the listener.
    """
    position_id = str(payload.get("positionId") or payload.get("id") or "")
    if not position_id:
        log.debug("position.updated without positionId: %s", payload)
        return

    quantity = _dec(payload.get("quantity", payload.get("qty")))

    # Closed position — drop from cache and don't push.
    if quantity == 0:
        _last_pushed_pnl.pop(position_id, None)
        return

    asset = _asset_symbol(payload.get("asset") or payload.get("symbol") or "")
    side = str(payload.get("positionSide") or payload.get("side") or "").lower()
    entry_price = _dec(payload.get("entryPrice", payload.get("avgEntryPrice")))
    mark_price = _dec(
        payload.get("markPrice", payload.get("mark", payload.get("price")))
    )
    unrealized_pnl = _dec(payload.get("unrealizedPnl", payload.get("uPnl")))
    notional = quantity * entry_price

    now = datetime.now(timezone.utc)
    if not _should_push_pnl(position_id, unrealized_pnl, notional, now):
        return

    text = _format_live_pnl(
        asset=asset,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        mark_price=mark_price,
        unrealized_pnl=unrealized_pnl,
    )
    try:
        await bot.send_message(
            chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN_V2
        )
        _last_pushed_pnl[position_id] = (now, unrealized_pnl)
    except Exception as exc:  # noqa: BLE001 — network / rate-limit errors
        log.error("Failed to send live PnL update: %s", exc)


# ---------------------------------------------------------------------------
# Fill-event waiter registry — used by the deferred-bracket fallback to place
# SL/TP only after the entry has actually filled, dodging the 13056
# ``conditional_order_requires_position_or_group`` race.
# ---------------------------------------------------------------------------
async def wait_for_fill(
    order_id: str, timeout: float = 90.0
) -> Optional[Dict[str, Any]]:
    """Resolve when a matching ``order.filled`` event arrives for *order_id*.

    Returns the event payload dict, or ``None`` on timeout. The caller is
    expected to schedule placement of dependent orders (SL/TP) in the
    resolving callback. Registration is idempotent — only one waiter per
    order id; concurrent callers share the same future.

    Also resolves on ``order.partially_filled``, ``order.triggered`` and
    ``position.opened`` so venue-specific event ordering doesn't block us.
    """
    loop = asyncio.get_event_loop()
    async with _WAITER_LOCK:
        fut = _FILL_WAITERS.get(order_id)
        if fut is None:
            fut = loop.create_future()
            _FILL_WAITERS[order_id] = fut
    try:
        return await asyncio.wait_for(fut, timeout=timeout)
    except asyncio.TimeoutError:
        return None
    finally:
        async with _WAITER_LOCK:
            # Only clear if this is still the future we registered — a
            # reconnect could have raced and replaced it.
            if _FILL_WAITERS.get(order_id) is fut:
                _FILL_WAITERS.pop(order_id, None)


def _payload_order_id(payload: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of an order id from an event payload.

    Different event kinds nest the id differently:
    - ``order.*`` usually has ``orderId`` at the top.
    - ``position.opened`` may carry ``entryOrderId`` or a nested ``order``.
    """
    for key in ("orderId", "entryOrderId", "id"):
        val = payload.get(key)
        if val:
            return str(val)
    nested = payload.get("order")
    if isinstance(nested, dict):
        for key in ("orderId", "id"):
            val = nested.get(key)
            if val:
                return str(val)
    return None


def _resolve_fill_waiter(event: str, payload: Dict[str, Any]) -> None:
    """Resolve any waiter registered for this event's order id.

    Runs BEFORE the Telegram-message formatting in :func:`_handle_message`
    so that a crash in message rendering can't leave the bracket-placement
    task hanging — the worst case is we set the future and the message
    fails to send.
    """
    if event not in _FILL_RESOLVING_EVENTS:
        return
    order_id = _payload_order_id(payload)
    if not order_id:
        return
    fut = _FILL_WAITERS.get(order_id)
    if fut is None or fut.done():
        return
    try:
        fut.set_result(payload)
    except asyncio.InvalidStateError:
        # Future was completed or cancelled by another resolver.
        pass


async def _handle_message(app: Any, chat_id: int, raw: Any) -> None:
    """Parse a single websocket frame and dispatch to Telegram if relevant."""
    bot = app.bot
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            log.debug("non-utf8 frame: %r", raw[:64])
            return

    try:
        msg = json.loads(raw)
    except (TypeError, ValueError):
        log.debug("non-json frame: %r", str(raw)[:200])
        return

    if not isinstance(msg, dict):
        return

    event_raw = msg.get("event") or msg.get("type") or msg.get("eventType") or ""
    event = _canonical_event(str(event_raw))
    if not event:
        log.debug("frame without event/type: %s", msg)
        return

    payload = _extract_payload(msg)

    # Resolve any bracket-placement waiter BEFORE the Telegram send so a
    # message-formatting crash cannot leave the deferred SL/TP task hanging.
    # The waiter registry is populated by handlers placing limit entries
    # that need to wait for a fill before attaching conditional legs.
    try:
        _resolve_fill_waiter(event, payload)
    except Exception as exc:  # noqa: BLE001 — never let a waiter bug crash ws
        log.debug("fill waiter resolver error: %s", exc)

    # Live PnL stream — only position.updated, and only when the toggle is on.
    if event == "position.updated":
        if app.bot_data.get("live_pnl_enabled"):
            await _maybe_push_live_pnl(bot, chat_id, payload)
        return

    text = _format_event(event, payload)
    if text is None:
        log.debug("ignoring unknown event: %s", event)
        return

    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except Exception as exc:  # noqa: BLE001 — network / rate-limit errors
        log.error("Failed to send ws event to telegram: %s", exc)


async def run_ws_listener(
    app: Any, chat_id: int, api_key: str, account_id: str
) -> None:
    """Connect to the Propr websocket and forward events to Telegram.

    Reconnects indefinitely with a 5-second backoff. The ``account_id`` is
    currently unused by the transport layer but kept in the signature so that
    callers can pass it if/when the API requires a subscribe frame. *app* is
    the ``telegram.ext.Application`` — the listener reads ``app.bot`` for
    sends and ``app.bot_data`` for the live-PnL toggle.
    """
    headers = {"X-API-Key": api_key}

    while True:
        try:
            log.info("Connecting to Propr websocket %s for account %s",
                     WS_URL, account_id)
            # websockets 12.0 uses ``extra_headers``.
            async with websockets.connect(
                WS_URL,
                extra_headers=headers,
                ping_interval=30,
                ping_timeout=20,
            ) as ws:
                log.info("Propr websocket connected")
                # Some deployments require a subscribe frame; the public docs
                # don't specify one, so we just listen. If the server closes
                # for lack of a subscription, add your subscribe message here.
                async for message in ws:
                    try:
                        await _handle_message(app, chat_id, message)
                    except Exception as exc:  # noqa: BLE001 — never die on a bad frame
                        log.exception("ws message handler error: %s", exc)
        except asyncio.CancelledError:
            log.info("ws listener cancelled")
            raise
        except InvalidStatusCode as exc:
            # PR #1 review finding #11 — handshake rejections for bad creds
            # must kill the process instead of retrying forever. Other 4xx/5xx
            # codes fall through to the generic reconnect branch. Review
            # finding 9 promotes this from ``raise SystemExit`` (which only
            # killed the asyncio task) to ``os._exit(1)`` after a best-effort
            # Telegram notice, so the whole bot exits on auth failure.
            status = getattr(exc, "status_code", None)
            if status in (401, 403):
                log.critical("ws handshake rejected with %s: %s", status, exc)
                # review finding 9: notify the authorized chat before forcing
                # interpreter exit; wrap the send in try/except so a failed
                # send can't block shutdown.
                try:
                    await app.bot.send_message(
                        chat_id=chat_id,
                        text="🛑 WS auth rejected — shutting down. Check PROPR_API_KEY.",
                    )
                except Exception as notify_exc:  # noqa: BLE001 — don't block exit
                    log.warning(
                        "failed to send ws auth-failure notice: %s", notify_exc
                    )
                # os._exit bypasses atexit / asyncio cleanup — required because
                # SystemExit inside an asyncio task only kills that task, not
                # the parent process.
                os._exit(1)
            log.warning("ws handshake returned %s; reconnecting in %ss",
                        status, RECONNECT_DELAY)
        except InvalidHandshake as exc:
            # Parent class — most variants don't carry a status_code, so we
            # just log and retry. PR #1 review finding #11.
            log.warning("ws handshake failed (%s); reconnecting in %ss",
                        exc, RECONNECT_DELAY)
        except (ConnectionClosed, WebSocketException, OSError) as exc:
            log.warning("ws disconnected (%s); reconnecting in %ss",
                        exc, RECONNECT_DELAY)
        except Exception as exc:  # noqa: BLE001
            log.exception("ws loop unexpected error: %s", exc)

        try:
            await asyncio.sleep(RECONNECT_DELAY)
        except asyncio.CancelledError:
            raise
