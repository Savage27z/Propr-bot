"""Telegram command & callback handlers for the Propr trading bot.

Every handler:

* Verifies that ``update.effective_chat.id`` matches ``TELEGRAM_CHAT_ID``.
* Wraps the body in ``try/except`` so a single error cannot crash the bot.
* Uses ``context.bot_data['propr']`` (a :class:`ProprClient`) and
  ``context.bot_data['account_id']`` which are injected in ``bot.py``.

Replies use MarkdownV2; :func:`escape_md` is used on anything user- or
API-derived.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import traceback
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import wraps
from typing import Any, Awaitable, Callable, List, Optional, Tuple

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes
from ulid import ULID

from analysis import (
    PENDING_INTENTS,
    PENDING_TRADES,
    VALID_INTERVALS,
    PendingIntent,
    PendingTrade,
    fetch_spot_price,
    parse_groq_recommendation,
    parse_trade_intent,
    run_analysis,
)
from pnl import build_snapshot, format_snapshot_markdown, render_snapshot_image
from propr import (
    ProprAPIError,
    ProprClient,
    asset_ticker,
    is_conditional_requires_position,
    max_leverage_for,
    normalize_asset,
)
from utils import escape_md, fmt_num
from ws_listener import wait_for_fill

log = logging.getLogger(__name__)

TELEGRAM_MAX_LEN = 4000  # safe under Telegram's 4096 hard cap


# ---------------------------------------------------------------------------
# Formatting helpers — ``escape_md`` and ``fmt_num`` live in :mod:`utils` now
# so :mod:`pnl` and :mod:`ws_listener` can share them without import cycles.
# ---------------------------------------------------------------------------
def _split_message(text: str, limit: int = TELEGRAM_MAX_LEN) -> List[str]:
    """Split *text* into chunks ≤ *limit* on line boundaries."""
    if len(text) <= limit:
        return [text]
    lines = text.split("\n")
    chunks: List[str] = []
    current = ""
    for line in lines:
        if len(line) > limit:
            # Single giant line — hard-split it.
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(line), limit):
                chunks.append(line[i:i + limit])
            continue
        candidate = line if not current else current + "\n" + line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------
HandlerFn = Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]


def authorized(func: HandlerFn) -> HandlerFn:
    """Restrict the handler to the configured Telegram chat id."""

    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        raw = os.getenv("TELEGRAM_CHAT_ID", "")
        try:
            allowed_id = int(raw)
        except ValueError:
            log.error("TELEGRAM_CHAT_ID is not a valid integer: %r", raw)
            if chat:
                await context.bot.send_message(chat_id=chat.id, text="⛔ Unauthorized")
            return
        if chat is None or chat.id != allowed_id:
            if chat:
                await context.bot.send_message(chat_id=chat.id, text="⛔ Unauthorized")
            log.warning("Unauthorized chat %s blocked from %s", chat.id if chat else "?", func.__name__)
            return

        user = update.effective_user
        args = getattr(context, "args", None)
        log.info(
            "handler=%s chat=%s user=%s args=%s",
            func.__name__,
            chat.id,
            user.id if user else "?",
            args,
        )
        return await func(update, context)

    return wrapper


# ---------------------------------------------------------------------------
# Argument / error helpers
# ---------------------------------------------------------------------------
def parse_args(
    context: ContextTypes.DEFAULT_TYPE, min_args: int, max_args: int
) -> Optional[List[str]]:
    """Return ``context.args`` if its length is in ``[min_args, max_args]``.

    Returns ``None`` when the count is out of range; the caller should then
    reply with a usage message.
    """
    args = list(context.args or [])
    if min_args <= len(args) <= max_args:
        return args
    return None


async def _reply(update: Update, text: str, **kwargs: Any) -> Any:
    """Send a MarkdownV2 reply to the current chat."""
    msg = update.effective_message
    if msg is None:
        return None
    return await msg.reply_text(
        text, parse_mode=ParseMode.MARKDOWN_V2, **kwargs
    )


async def _report_error(update: Update, exc: Exception) -> None:
    """Translate *exc* into a friendly MarkdownV2 message and log it."""
    if isinstance(exc, ProprAPIError):
        text = f"❌ API error: {escape_md(exc)}"
    else:
        text = f"❌ Unexpected error: {escape_md(type(exc).__name__)}: {escape_md(exc)}"
    log.error("handler error: %s\n%s", exc, traceback.format_exc())
    try:
        await _reply(update, text)
    except Exception as reply_exc:  # noqa: BLE001
        log.error("failed to send error reply: %s", reply_exc)


def _get_client(context: ContextTypes.DEFAULT_TYPE) -> ProprClient:
    client = context.bot_data.get("propr")
    if client is None:
        raise RuntimeError("ProprClient not initialized in bot_data")
    return client


def _get_account_id(context: ContextTypes.DEFAULT_TYPE) -> Optional[str]:
    return context.bot_data.get("account_id")


async def _ensure_account_id(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Return the cached accountId or fetch and cache it."""
    existing = _get_account_id(context)
    if existing:
        return existing
    client = _get_client(context)
    account_id = await client.get_active_account_id()
    context.bot_data["account_id"] = account_id
    return account_id


def _parse_qty(raw: str) -> Decimal:
    """Parse a quantity string into Decimal, raising ``ValueError`` on junk."""
    try:
        d = Decimal(raw.replace(",", "").replace("_", ""))
    except (InvalidOperation, ValueError):
        raise ValueError(f"invalid quantity: {raw!r}")
    if d <= 0:
        raise ValueError("quantity must be positive")
    return d


def _parse_price(raw: str) -> Decimal:
    """Parse a price string into Decimal."""
    try:
        d = Decimal(raw.replace(",", "").replace("_", ""))
    except (InvalidOperation, ValueError):
        raise ValueError(f"invalid price: {raw!r}")
    if d <= 0:
        raise ValueError("price must be positive")
    return d


def _pos_asset_upper(p: dict) -> str:
    # review finding 4: delegate to the canonical asset_ticker helper so
    # OIL→CL (and any other xyz: alias) compares the same way downstream.
    return asset_ticker(p.get("asset", ""))


def _pos_side(p: dict) -> str:
    return str(p.get("positionSide") or p.get("side") or "").lower()


def _extract_order_id(response: Any) -> Optional[str]:
    """Pull the server-assigned ``orderId`` out of a ``place_order`` response.

    Propr returns orders in one of two envelopes — either ``data[0]`` when
    the caller submitted an ``orders`` array, or a flat object with
    ``orderId``. Review finding 2 needs this id to roll back a naked entry
    after an SL failure by cancelling the entry before closing the fill.
    """
    # review finding 2: normalise both response shapes into a single id.
    if not isinstance(response, dict):
        return None
    direct = response.get("orderId") or response.get("id")
    if direct:
        return str(direct)
    data = response.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            candidate = first.get("orderId") or first.get("id")
            if candidate:
                return str(candidate)
    if isinstance(data, dict):
        candidate = data.get("orderId") or data.get("id")
        if candidate:
            return str(candidate)
    return None


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
@authorized
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/start`` — welcome, health check, basic account balance."""
    try:
        client = _get_client(context)
        health = await client.health()
        me = await client.me()
        account_id = await _ensure_account_id(context)

        balance = "?"
        if isinstance(me, dict):
            balance = (
                me.get("balance")
                or me.get("equity")
                or (me.get("account") or {}).get("balance")
                or "?"
            )
        if balance == "?":
            # fall back to positions endpoint account summary
            try:
                positions = await client.get_positions(account_id)
                if positions and isinstance(positions[0].get("accountBalance"), (int, float, str)):
                    balance = positions[0]["accountBalance"]
            except ProprAPIError:
                pass

        status = health.get("status") if isinstance(health, dict) else "ok"
        msg = (
            "👋 *Welcome to Propr Bot*\n\n"
            f"API health: `{escape_md(status)}`\n"
            f"Account: `{escape_md(account_id)}`\n"
            f"Balance: *{escape_md(fmt_num(balance))}* USDC\n\n"
            "Type /help for the full command list\\."
        )
        await _reply(update, msg)
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/status`` — API health, challenge status, P&L summary."""
    try:
        client = _get_client(context)
        account_id = await _ensure_account_id(context)

        health, attempts, positions, trades = await asyncio.gather(
            client.health(),
            client.get_challenge_attempts(),
            client.get_positions(account_id),
            client.get_trades(account_id, limit=50),
            return_exceptions=True,
        )

        def _ok(v: Any) -> Any:
            return v if not isinstance(v, Exception) else None

        health = _ok(health) or {}
        attempts = _ok(attempts) or []
        positions = _ok(positions) or []
        trades = _ok(trades) or []

        attempt = attempts[0] if attempts else {}
        status_val = health.get("status", "unknown") if isinstance(health, dict) else "unknown"
        attempt_status = attempt.get("status", "?") if isinstance(attempt, dict) else "?"

        realized = sum((Decimal(str(t.get("pnl", 0) or 0)) for t in trades), Decimal(0))
        unrealized = sum(
            (
                Decimal(str(p.get("unrealizedPnl", p.get("uPnl", 0) or 0)))
                for p in positions
            ),
            Decimal(0),
        )

        text = (
            "📊 *Status*\n\n"
            f"API: `{escape_md(status_val)}`\n"
            f"Challenge: `{escape_md(attempt_status)}`\n"
            f"Account: `{escape_md(account_id)}`\n"
            f"Realized PnL \\(recent\\): *{escape_md(fmt_num(realized))}* USDC\n"
            f"Unrealized PnL: *{escape_md(fmt_num(unrealized))}* USDC\n"
            f"Open positions: `{len(positions)}`"
        )
        await _reply(update, text)
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def positions_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/positions`` — list open positions."""
    try:
        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        positions = await client.get_positions(account_id)
        if not positions:
            await _reply(update, "📂 No open positions\\.")
            return

        lines = ["📈 *Open Positions*"]
        for p in positions:
            asset = p.get("asset", "?")
            side = _pos_side(p) or "?"
            qty = p.get("quantity", p.get("qty", "?"))
            entry = p.get("entryPrice", p.get("avgEntryPrice", "?"))
            mark = p.get("markPrice", p.get("mark", "?"))
            upnl = p.get("unrealizedPnl", p.get("uPnl", "?"))
            lev = p.get("leverage", "?")
            lines.append(
                f"• `{escape_md(asset)}` "
                f"{escape_md(side)} "
                f"`{escape_md(fmt_num(qty, 6))}`"
                f" \\| entry *{escape_md(fmt_num(entry))}*"
                f" \\| mark `{escape_md(fmt_num(mark))}`"
                f" \\| uPnL *{escape_md(fmt_num(upnl))}*"
                f" \\| `{escape_md(lev)}x`"
            )
        await _reply(update, "\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def orders_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/orders`` — list open orders."""
    try:
        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        orders = await client.get_open_orders(account_id)
        if not orders:
            await _reply(update, "📭 No open orders\\.")
            return

        lines = ["📝 *Open Orders*"]
        for o in orders:
            oid = o.get("id", o.get("orderId", "?"))
            side = o.get("side", "?")
            otype = o.get("type", "?")
            asset = o.get("asset", "?")
            qty = o.get("quantity", o.get("qty", "?"))
            price = o.get("price", o.get("triggerPrice", "?"))
            lines.append(
                f"• `{escape_md(oid)}` "
                f"{escape_md(otype)} {escape_md(side)} "
                f"`{escape_md(fmt_num(qty, 6))}` {escape_md(asset)} "
                f"@ *{escape_md(fmt_num(price))}*"
            )
        await _reply(update, "\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def trades_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/trades [n]`` — last *n* trades (default 5)."""
    try:
        args = parse_args(context, 0, 1)
        if args is None:
            await _reply(update, "Usage: `/trades [n]`")
            return
        limit = 5
        if args:
            try:
                limit = max(1, min(50, int(args[0])))
            except ValueError:
                await _reply(update, "⚠️ n must be an integer")
                return

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        trades = await client.get_trades(account_id, limit=limit)
        if not trades:
            await _reply(update, "📂 No trades yet\\.")
            return

        lines = [f"🧾 *Last {limit} Trades*"]
        for t in trades:
            ts = t.get("createdAt") or t.get("timestamp") or t.get("time") or ""
            asset = t.get("asset", "?")
            side = t.get("side", "?")
            qty = t.get("quantity", t.get("qty", "?"))
            price = t.get("price", t.get("avgPrice", "?"))
            pnl = t.get("pnl", "?")
            lines.append(
                f"• `{escape_md(ts)}` "
                f"{escape_md(side)} `{escape_md(fmt_num(qty, 6))}` "
                f"{escape_md(asset)} @ *{escape_md(fmt_num(price))}* "
                f"\\| PnL {escape_md(fmt_num(pnl))}"
            )
        await _reply(update, "\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/buy <asset> <qty> [price]`` — market if no price, limit otherwise."""
    try:
        args = parse_args(context, 2, 3)
        if args is None:
            await _reply(update, "Usage: `/buy <asset> <qty> [price]`")
            return
        asset = args[0].upper()
        qty = _parse_qty(args[1])
        price: Optional[Decimal] = None
        if len(args) == 3:
            price = _parse_price(args[2])

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        if price is None:
            result = await client.market_buy(account_id, asset, qty)
            mode = "market"
        else:
            result = await client.limit_buy(account_id, asset, qty, price)
            mode = f"limit @ {fmt_num(price)}"

        await _reply(
            update,
            f"🟢 *Buy* `{escape_md(fmt_num(qty, 6))}` {escape_md(asset)} "
            f"\\({escape_md(mode)}\\) submitted\\.",
        )
        log.debug("buy result: %s", result)
    except ValueError as exc:
        await _reply(update, f"⚠️ {escape_md(exc)}")
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def sell_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/sell <asset> <qty> [price]`` — always reduce-only."""
    try:
        args = parse_args(context, 2, 3)
        if args is None:
            await _reply(update, "Usage: `/sell <asset> <qty> [price]`")
            return
        asset = args[0].upper()
        qty = _parse_qty(args[1])
        price: Optional[Decimal] = None
        if len(args) == 3:
            price = _parse_price(args[2])

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        if price is None:
            await client.market_sell(account_id, asset, qty, reduce_only=True)
            mode = "market"
        else:
            await client.limit_sell(account_id, asset, qty, price, reduce_only=True)
            mode = f"limit @ {fmt_num(price)}"

        await _reply(
            update,
            f"🔴 *Sell* `{escape_md(fmt_num(qty, 6))}` {escape_md(asset)} "
            f"\\({escape_md(mode)}, reduce\\-only\\) submitted\\.",
        )
    except ValueError as exc:
        await _reply(update, f"⚠️ {escape_md(exc)}")
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def close_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/close <asset>`` — fully close current position on *asset*."""
    try:
        args = parse_args(context, 1, 1)
        if args is None:
            await _reply(update, "Usage: `/close <asset>`")
            return
        asset = args[0].upper()
        # review finding 4: compare positions via the venue ticker so aliases
        # like OIL→CL route through cleanly; keep ``asset`` for user replies.
        asset_norm_ticker = asset_ticker(normalize_asset(asset))

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        positions = await client.get_positions(account_id)
        target = next(
            (p for p in positions if _pos_asset_upper(p) == asset_norm_ticker), None
        )
        if target is None:
            await _reply(update, f"⚠️ No open position on `{escape_md(asset)}`")
            return
        side = _pos_side(target)
        if side not in ("long", "short"):
            await _reply(update, "⚠️ Could not determine position side\\.")
            return

        await client.close_position(account_id, asset, side)
        await _reply(
            update,
            f"🧹 Closing *{escape_md(side.upper())}* position on {escape_md(asset)}\\.",
        )
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/cancel <orderId>`` — cancel a single order."""
    try:
        args = parse_args(context, 1, 1)
        if args is None:
            await _reply(update, "Usage: `/cancel <orderId>`")
            return
        order_id = args[0]
        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        result = await client.cancel_order(account_id, order_id)
        if isinstance(result, dict) and result.get("status") == "already_done":
            await _reply(update, f"ℹ️ Order `{escape_md(order_id)}` already filled or cancelled\\.")
        else:
            await _reply(update, f"✅ Cancelled `{escape_md(order_id)}`")
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def cancelall_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/cancelall`` — cancel every open order, tolerating 400s."""
    try:
        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        orders = await client.get_open_orders(account_id)
        if not orders:
            await _reply(update, "📭 No open orders\\.")
            return

        results = await asyncio.gather(
            *[
                client.cancel_order(
                    account_id, str(o.get("id") or o.get("orderId") or "")
                )
                for o in orders
                if (o.get("id") or o.get("orderId"))
            ],
            return_exceptions=True,
        )
        ok = sum(
            1
            for r in results
            if not isinstance(r, Exception)
        )
        failed = sum(1 for r in results if isinstance(r, Exception))
        await _reply(
            update,
            f"🧹 Cancel\\-all: `{ok}` ok, `{failed}` failed",
        )
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def sl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/sl <asset> <triggerPrice>`` — stop-loss on the current position.

    Picks whichever side (long or short) is currently open on *asset* —
    previously rejected shorts outright (PR #1 review finding #9).
    """
    try:
        args = parse_args(context, 2, 2)
        if args is None:
            await _reply(update, "Usage: `/sl <asset> <triggerPrice>`")
            return
        asset = args[0].upper()
        # review finding 4: resolve through normalize→asset_ticker so alias
        # tokens (e.g. OIL→CL) match the position asset row reliably.
        asset_norm_ticker = asset_ticker(normalize_asset(asset))
        trigger = _parse_price(args[1])

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        positions = await client.get_positions(account_id)
        target = next(
            (p for p in positions if _pos_asset_upper(p) == asset_norm_ticker),
            None,
        )
        if target is None:
            await _reply(update, f"⚠️ No open position on `{escape_md(asset)}`")
            return
        side = _pos_side(target)
        if side not in ("long", "short"):
            await _reply(update, "⚠️ Could not determine position side\\.")
            return
        qty = target.get("quantity", target.get("qty"))
        await client.stop_loss(account_id, asset, qty, trigger, side)
        await _reply(
            update,
            f"🛡 Stop\\-loss set on {escape_md(side.upper())} {escape_md(asset)} "
            f"at *{escape_md(fmt_num(trigger))}*",
        )
    except ValueError as exc:
        await _reply(update, f"⚠️ {escape_md(exc)}")
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def tp_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/tp <asset> <triggerPrice>`` — take-profit on the current position.

    Picks whichever side (long or short) is currently open on *asset* —
    previously rejected shorts outright (PR #1 review finding #9).
    """
    try:
        args = parse_args(context, 2, 2)
        if args is None:
            await _reply(update, "Usage: `/tp <asset> <triggerPrice>`")
            return
        asset = args[0].upper()
        # review finding 4: same alias-safe comparison as /sl.
        asset_norm_ticker = asset_ticker(normalize_asset(asset))
        trigger = _parse_price(args[1])

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        positions = await client.get_positions(account_id)
        target = next(
            (p for p in positions if _pos_asset_upper(p) == asset_norm_ticker),
            None,
        )
        if target is None:
            await _reply(update, f"⚠️ No open position on `{escape_md(asset)}`")
            return
        side = _pos_side(target)
        if side not in ("long", "short"):
            await _reply(update, "⚠️ Could not determine position side\\.")
            return
        qty = target.get("quantity", target.get("qty"))
        await client.take_profit(account_id, asset, qty, trigger, side)
        await _reply(
            update,
            f"🎯 Take\\-profit set on {escape_md(side.upper())} {escape_md(asset)} "
            f"at *{escape_md(fmt_num(trigger))}*",
        )
    except ValueError as exc:
        await _reply(update, f"⚠️ {escape_md(exc)}")
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def leverage_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/leverage <asset> <n>`` — change leverage subject to caps."""
    try:
        args = parse_args(context, 2, 2)
        if args is None:
            await _reply(update, "Usage: `/leverage <asset> <n>`")
            return
        asset = args[0].upper()
        try:
            lev = int(args[1])
        except ValueError:
            await _reply(update, "⚠️ Leverage must be an integer")
            return
        cap = max_leverage_for(asset)
        if lev > cap or lev <= 0:
            await _reply(
                update,
                f"⚠️ Leverage for {escape_md(asset)} must be 1\\-{cap}x",
            )
            return

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        await client.set_leverage(account_id, asset, lev)
        await _reply(
            update,
            f"⚙️ Leverage for {escape_md(asset)} set to *{lev}x*",
        )
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def analysis_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/analysis <asset> [timeframe] [direction]`` — expert AI analysis."""
    try:
        args = parse_args(context, 1, 3)
        if args is None:
            await _reply(update, "Usage: `/analysis <asset> [timeframe] [direction]`")
            return

        asset = args[0].upper()
        timeframe = args[1].lower() if len(args) >= 2 else "1h"
        direction = args[2].lower() if len(args) >= 3 else None

        # PR #1 review finding #10 — gate on a whitelist so we don't kick off a
        # Groq call with a bogus interval that Binance will reject.
        if timeframe not in VALID_INTERVALS:
            await _reply(
                update,
                "⚠️ Unsupported timeframe\\. Use: 15m, 1h, 4h, 1d",
            )
            return

        await _run_analysis_flow(
            context,
            chat_id=update.effective_chat.id,
            asset=asset,
            timeframe=timeframe,
            direction=direction,
            intro_reply=update.effective_message,
        )
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


async def _run_analysis_flow(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    asset: str,
    timeframe: str,
    direction: Optional[str],
    intro_reply: Any,
) -> None:
    """Shared ``/analysis`` pipeline — reused by the NL ``analysis`` intent.

    Sends the intro line, runs the Groq analysis with a typing indicator,
    chunks the response, attaches the Execute/Skip keyboard when the parsed
    recommendation is auto-executable, and stores a :class:`PendingTrade`.
    """
    typing_task: Optional[asyncio.Task] = None
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        if intro_reply is not None:
            await intro_reply.reply_text(
                f"🔍 Analyzing {escape_md(asset)}\\.\\.\\.",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"🔍 Analyzing {escape_md(asset)}\\.\\.\\.",
                parse_mode=ParseMode.MARKDOWN_V2,
            )

        async def _typing_loop() -> None:
            try:
                while True:
                    await asyncio.sleep(4)
                    await context.bot.send_chat_action(
                        chat_id=chat_id, action=ChatAction.TYPING
                    )
            except asyncio.CancelledError:
                return

        typing_task = asyncio.create_task(_typing_loop())

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        result = await run_analysis(
            client, account_id, asset, timeframe=timeframe, direction=direction
        )

        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass
        typing_task = None

        raw = result["raw"]
        parsed = result["parsed"]
        fetched_at = result["market_fetched_at"]

        pending_id = str(ULID())
        footer = f"\n\n📡 Market data fetched at {fetched_at}"
        full_text = raw + footer
        chunks = _split_message(full_text, TELEGRAM_MAX_LEN)

        # PR #1 review finding #14 — only show the Execute/Skip keyboard and
        # store a PENDING_TRADES entry when the parse yielded an auto-executable
        # trade. Non-executable analyses go out as text only.
        executable = bool(parsed.get("executable"))
        keyboard = InlineKeyboardMarkup(
            [[
                InlineKeyboardButton(
                    "✅ Execute Trade", callback_data=f"exec_trade:{pending_id}"
                ),
                InlineKeyboardButton(
                    "❌ Skip", callback_data=f"skip_trade:{pending_id}"
                ),
            ]]
        ) if executable else None

        last_message_id: Optional[int] = None
        for idx, chunk in enumerate(chunks):
            is_last = idx == len(chunks) - 1
            escaped = escape_md(chunk)
            sent = await context.bot.send_message(
                chat_id=chat_id,
                text=escaped,
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=keyboard if is_last else None,
            )
            last_message_id = sent.message_id

        if executable:
            PENDING_TRADES[pending_id] = PendingTrade(
                pending_id=pending_id,
                chat_id=chat_id,
                asset=asset,
                timeframe=timeframe,
                raw_analysis=raw,
                parsed=parsed,
                created_at=datetime.now(timezone.utc),
                message_id=last_message_id,
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "ℹ️ This analysis is not auto\\-executable "
                    "\\(missing direction, size, SL, or TP1\\)\\. "
                    "Review and place manually with /buy, /sell, /sl, /tp\\."
                ),
                parse_mode=ParseMode.MARKDOWN_V2,
            )
    finally:
        if typing_task and not typing_task.done():
            typing_task.cancel()


# ---------------------------------------------------------------------------
# Callback handler for Execute / Skip buttons
# ---------------------------------------------------------------------------
def _callback_authorized(query_chat_id: int) -> bool:
    raw = os.getenv("TELEGRAM_CHAT_ID", "")
    try:
        return query_chat_id == int(raw)
    except ValueError:
        return False


async def callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle ``exec_trade`` / ``skip_trade`` and ``exec_intent`` / ``skip_intent`` clicks.

    - ``exec_trade`` / ``skip_trade`` come from the ``/analysis`` flow.
    - ``exec_intent`` / ``skip_intent`` come from natural-language ``/trade``
      confirmations.
    Routing is strictly by prefix so the two stores never collide.
    """
    query = update.callback_query
    if query is None:
        return

    chat = update.effective_chat
    if chat is None or not _callback_authorized(chat.id):
        await query.answer("⛔ Unauthorized", show_alert=False)
        return

    data = query.data or ""
    action, _, pending_id = data.partition(":")

    if action in ("exec_trade", "skip_trade"):
        await _handle_analysis_callback(context, query, chat.id, action, pending_id)
        return
    if action in ("exec_intent", "skip_intent"):
        await _handle_intent_callback(context, query, chat.id, action, pending_id)
        return

    await query.answer("Unknown action")


async def _handle_analysis_callback(
    context: ContextTypes.DEFAULT_TYPE,
    query: Any,
    chat_id: int,
    action: str,
    pending_id: str,
) -> None:
    """Route a callback for the ``/analysis`` confirmation keyboard."""
    pending = PENDING_TRADES.get(pending_id)
    if pending is None:
        await query.answer("Expired")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception as exc:  # noqa: BLE001
            log.debug("edit_message_reply_markup on expired failed: %s", exc)
        return

    if action == "skip_trade":
        PENDING_TRADES.pop(pending_id, None)
        await query.answer("Skipped")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
            await context.bot.send_message(chat_id=chat_id, text="❎ Trade skipped.")
        except Exception as exc:  # noqa: BLE001
            log.debug("skip cleanup failed: %s", exc)
        return

    await query.answer("Executing…")
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception as exc:  # noqa: BLE001
        log.debug("edit_message_reply_markup failed: %s", exc)

    PENDING_TRADES.pop(pending_id, None)

    # Re-parse from raw text in case the stored dict is stale — spec requires it.
    parsed = parse_groq_recommendation(pending.raw_analysis) or pending.parsed

    if parsed.get("direction") == "no_trade" or not parsed.get("executable"):
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "⚠️ Could not auto-parse trade parameters. "
                "Review analysis and place manually with /buy or /sell."
            ),
        )
        return

    await _execute_parsed_trade(context, chat_id, pending.asset, parsed)


async def _execute_parsed_trade(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    asset: str,
    parsed: dict,
) -> None:
    """Turn a parsed Groq recommendation into live Propr orders.

    Delegates to :func:`_execute_open_with_brackets` so the entry+SL+TP
    placement goes through the atomic-batch-with-deferred-fallback path
    that dodges the 13056
    ``conditional_order_requires_position_or_group`` race (previously
    this function placed entry, SL, and TP sequentially — which
    reliably tripped the race on limit entries).
    """
    client = _get_client(context)
    try:
        account_id = await _ensure_account_id(context)
    except Exception as exc:  # noqa: BLE001
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Execution failed at step account lookup: {exc}",
        )
        return

    direction: str = parsed["direction"]
    quantity: Decimal = parsed["quantity"]
    stop_loss: Decimal = parsed["stop_loss"]
    tp1: Decimal = parsed["take_profit_1"]
    tp2: Optional[Decimal] = parsed.get("take_profit_2")
    entry_price: Optional[Decimal] = parsed.get("entry_price")
    leverage: Optional[int] = parsed.get("leverage")
    order_type: str = parsed.get("order_type") or "market"

    position_side = "long" if direction == "long" else "short"
    take_profits: List[Decimal] = [tp1] + ([tp2] if tp2 is not None else [])

    ok, summary = await _execute_open_with_brackets(
        bot=context.bot,
        chat_id=chat_id,
        bot_data=context.application.bot_data,
        client=client,
        account_id=account_id,
        asset=asset,
        side=position_side,
        order_type=order_type,
        quantity=quantity,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profits=take_profits,
        leverage=leverage,
    )

    if ok:
        await context.bot.send_message(
            chat_id=chat_id, text=f"🚀 {summary}"
        )
    else:
        await context.bot.send_message(
            chat_id=chat_id, text=f"❌ Execution failed: {summary}"
        )


def _half_qty(qty: Decimal) -> Decimal:
    """Split *qty* in half, rounded down to a reasonable precision."""
    half = qty / Decimal(2)
    # Preserve up to 8 decimal places to handle small-lot crypto sizes.
    return half.quantize(Decimal("0.00000001"))


# ---------------------------------------------------------------------------
# Atomic entry + brackets helper
#
# Unifies the auto-exec paths in ``/analysis`` and the NL ``open`` intent so
# both dodge the ``conditional_order_requires_position_or_group`` (13056)
# race. Strategy: try a single batched POST first (cheap + atomic on the
# Propr side); if the conditionals get rejected with 13056 — or we're on a
# limit entry that fills later — fall back to placing just the entry and
# deferring SL/TP to a background task that awaits the ws
# ``order.filled``/``position.opened`` event.
# ---------------------------------------------------------------------------
_BRACKETS_WAIT_TIMEOUT = 90.0  # seconds to wait for the entry fill event


def _build_entry_leg(
    *,
    account_id: str,
    asset: str,
    side: str,
    taker_side: str,
    order_type: str,
    quantity: Decimal,
    entry_price: Optional[Decimal],
) -> dict:
    """Shape the entry order for ``place_batch`` / ``place_order``."""
    leg: dict = {
        "accountId": account_id,
        "asset": asset,
        "type": order_type,
        "side": taker_side,
        "positionSide": side,
        "quantity": quantity,
        "reduceOnly": False,
    }
    if order_type == "limit" and entry_price is not None:
        leg["timeInForce"] = "GTC"
        leg["price"] = entry_price
    else:
        leg["timeInForce"] = "IOC"
    return leg


def _build_sl_leg(
    *,
    account_id: str,
    asset: str,
    side: str,
    opposite_side: str,
    quantity: Decimal,
    stop_loss: Decimal,
) -> dict:
    return {
        "accountId": account_id,
        "asset": asset,
        "type": "stop_market",
        "side": opposite_side,
        "positionSide": side,
        "timeInForce": "GTC",
        "quantity": quantity,
        "triggerPrice": stop_loss,
        "reduceOnly": True,
    }


def _build_tp_leg(
    *,
    account_id: str,
    asset: str,
    side: str,
    opposite_side: str,
    quantity: Decimal,
    trigger: Decimal,
) -> dict:
    return {
        "accountId": account_id,
        "asset": asset,
        "type": "take_profit_market",
        "side": opposite_side,
        "positionSide": side,
        "timeInForce": "GTC",
        "quantity": quantity,
        "triggerPrice": trigger,
        "reduceOnly": True,
    }


def _tp_qtys(quantity: Decimal, n_tps: int) -> List[Decimal]:
    """Return per-leg quantities for *n_tps* take-profit orders.

    One TP → full qty. Two TPs → half each, with the remainder going on TP2
    so we never over-allocate vs. the position size.
    """
    if n_tps <= 1:
        return [quantity]
    half = _half_qty(quantity)
    return [half, quantity - half]


def _format_summary(
    *,
    direction_upper: str,
    quantity: Decimal,
    asset: str,
    entry_desc: str,
    stop_loss: Optional[Decimal],
    take_profits: List[Decimal],
) -> str:
    parts = [f"{direction_upper} {fmt_num(quantity, 6)} {asset} @ {entry_desc}"]
    if stop_loss is not None:
        parts.append(f"SL: {fmt_num(stop_loss)}")
    for i, tp in enumerate(take_profits, start=1):
        parts.append(f"TP{i}: {fmt_num(tp)}")
    return " | ".join(parts)


async def _maybe_set_leverage(
    client: ProprClient, account_id: str, asset: str, leverage: Optional[int]
) -> None:
    """Set leverage only if it differs from the current margin config.

    Matches the behaviour of the pre-refactor sequential paths: avoids
    tripping Propr's rate limits on no-op PUTs.
    """
    if leverage is None or leverage <= 0:
        return
    current_cfg = await client.get_margin_config(account_id, asset)
    if (
        isinstance(current_cfg, dict)
        and "data" in current_cfg
        and isinstance(current_cfg["data"], dict)
    ):
        current_cfg = current_cfg["data"]
    current_lev: Any = None
    if isinstance(current_cfg, dict):
        current_lev = current_cfg.get("leverage")
    try:
        current_lev_int = (
            int(float(current_lev)) if current_lev is not None else None
        )
    except (TypeError, ValueError):
        current_lev_int = None
    if current_lev_int != leverage:
        await client.set_leverage(account_id, asset, leverage)


async def _rollback_naked_entry(
    *,
    bot: Any,
    chat_id: int,
    client: ProprClient,
    account_id: str,
    asset: str,
    side: str,
    order_type: str,
    entry_order_id: Optional[str],
    trigger_exc: Exception,
) -> str:
    """Cancel an unfilled limit + close any realised fill, returning a summary.

    Honors the ``404 == clean rollback`` rule used elsewhere: if close
    returns 404 AND the entry was a market (or we already cancelled the
    limit), nothing actually filled so it's a clean rollback. On any other
    close failure we push a loud "UNPROTECTED" alert and re-raise via the
    returned summary so the caller can decide how to escalate.
    """
    cancel_err: Optional[Exception] = None
    cancelled = False
    if order_type == "limit" and entry_order_id:
        try:
            await client.cancel_order(account_id, entry_order_id)
            cancelled = True
        except Exception as c_exc:  # noqa: BLE001
            cancel_err = c_exc
    closed = False
    try:
        await client.close_position(account_id, asset, side)
        closed = True
    except ProprAPIError as close_exc:
        if close_exc.status == 404 and (cancelled or order_type == "market"):
            # Nothing to close — entry never filled. Clean rollback.
            pass
        else:
            cancel_reason = f" | cancel={cancel_err}" if cancel_err else ""
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"🚨 SL placement failed AND close failed. "
                    f"OPEN POSITION UNPROTECTED. Manually close {asset} now. "
                    f"Reasons: SL={trigger_exc} | close={close_exc}{cancel_reason}"
                ),
            )
            return f"MANUAL INTERVENTION NEEDED ({trigger_exc})"
    except Exception as close_exc:  # noqa: BLE001
        cancel_reason = f" | cancel={cancel_err}" if cancel_err else ""
        await bot.send_message(
            chat_id=chat_id,
            text=(
                f"🚨 SL placement failed AND close failed. "
                f"OPEN POSITION UNPROTECTED. Manually close {asset} now. "
                f"Reasons: SL={trigger_exc} | close={close_exc}{cancel_reason}"
            ),
        )
        return f"MANUAL INTERVENTION NEEDED ({trigger_exc})"
    if cancelled and not closed:
        return "entry cancelled (nothing filled)"
    if cancelled and closed:
        return "cancelled + partial fill closed"
    if closed:
        return "closed"
    return "no position open"


async def _place_brackets_after_fill(
    *,
    bot: Any,
    chat_id: int,
    client: ProprClient,
    account_id: str,
    asset: str,
    side: str,
    order_type: str,
    entry_order_id: Optional[str],
    quantity: Decimal,
    stop_loss: Optional[Decimal],
    take_profits: List[Decimal],
) -> None:
    """Background task body: wait for the entry fill then attach SL/TP.

    Runs as a ``asyncio.create_task`` from :func:`_execute_open_with_brackets`
    so the user's confirmation message returns instantly while the brackets
    attach asynchronously. On SL failure the existing rollback sequence
    fires; TP failures are non-fatal (SL already protects the position).
    """
    if not entry_order_id:
        log.warning(
            "brackets task skipped: no entry_order_id for %s %s", side, asset
        )
        await bot.send_message(
            chat_id=chat_id,
            text=(
                f"⚠️ {asset} entry placed but no orderId returned — brackets "
                "not attached. Use /sl and /tp manually."
            ),
        )
        return

    try:
        fill_payload = await wait_for_fill(
            entry_order_id, timeout=_BRACKETS_WAIT_TIMEOUT
        )
    except Exception as exc:  # noqa: BLE001 — ws listener not running etc.
        # Defensive fallback: if the ws waiter blew up, sleep briefly and try
        # placement anyway. The user still gets SL/TP, just with less
        # determinism than the event-driven path.
        log.warning(
            "wait_for_fill raised, falling back to 3s sleep: %s", exc
        )
        await asyncio.sleep(3)
        fill_payload = {"fallback": True}

    if fill_payload is None:
        # Timeout — the entry didn't fill within _BRACKETS_WAIT_TIMEOUT.
        await bot.send_message(
            chat_id=chat_id,
            text=(
                f"⚠️ {asset} entry didn't fill within "
                f"{int(_BRACKETS_WAIT_TIMEOUT)}s — brackets not attached. "
                "Use /sl and /tp manually."
            ),
        )
        return

    opposite_side = "sell" if side == "long" else "buy"
    # Step 1: SL — critical. If it fails, rollback the now-live entry.
    if stop_loss is not None and stop_loss > 0:
        try:
            await client.stop_loss(
                account_id, asset, quantity, stop_loss, side
            )
        except Exception as sl_exc:  # noqa: BLE001
            summary = await _rollback_naked_entry(
                bot=bot,
                chat_id=chat_id,
                client=client,
                account_id=account_id,
                asset=asset,
                side=side,
                order_type=order_type,
                entry_order_id=entry_order_id,
                trigger_exc=sl_exc,
            )
            await bot.send_message(
                chat_id=chat_id,
                text=f"❌ SL placement failed; {summary}. Reason: {sl_exc}",
            )
            return

    # Step 2: TP(s) — non-fatal; user keeps SL protection on any failure.
    tp_msgs: List[str] = []
    tp_warnings: List[str] = []
    if take_profits:
        qtys = _tp_qtys(quantity, len(take_profits))
        for idx, (leg_qty, tp_trigger) in enumerate(
            zip(qtys, take_profits), start=1
        ):
            try:
                await client.take_profit(
                    account_id, asset, leg_qty, tp_trigger, side
                )
                tp_msgs.append(f"TP{idx} {fmt_num(tp_trigger)}")
            except Exception as tp_exc:  # noqa: BLE001
                tp_warnings.append(
                    f"⚠️ TP{idx} failed: {tp_exc} — SL is still active"
                )

    # Step 3: single follow-up message summarising what attached.
    parts: List[str] = []
    if stop_loss is not None:
        parts.append(f"SL {fmt_num(stop_loss)}")
    parts.extend(tp_msgs)
    if parts:
        await bot.send_message(
            chat_id=chat_id, text="🛡 Brackets placed: " + " | ".join(parts)
        )
    for warning in tp_warnings:
        await bot.send_message(chat_id=chat_id, text=warning)


def _track_bracket_task(bot_data: dict, task: asyncio.Task) -> None:
    """Retain a reference to *task* so it isn't GC'd mid-run."""
    tasks: List[asyncio.Task] = bot_data.setdefault("bracket_tasks", [])
    tasks.append(task)
    # Drop done tasks opportunistically so the list doesn't grow unbounded.
    bot_data["bracket_tasks"] = [t for t in tasks if not t.done()]


async def _execute_open_with_brackets(
    *,
    bot: Any,
    chat_id: int,
    bot_data: dict,
    client: ProprClient,
    account_id: str,
    asset: str,
    side: str,
    order_type: str,
    quantity: Decimal,
    entry_price: Optional[Decimal],
    stop_loss: Optional[Decimal],
    take_profits: List[Decimal],
    leverage: Optional[int] = None,
) -> Tuple[bool, str]:
    """Atomic entry + SL + TP placement with deferred fallback.

    Strategy (dodges the 13056
    ``conditional_order_requires_position_or_group`` race):

    1. If *leverage* differs from the current config, set it.
    2. If neither SL nor TPs are requested, place only the entry — there's
       no conditional to race against.
    3. Otherwise try a single batched POST (entry + SL + TP legs). The
       Propr server groups them against the resulting position if it can,
       which sidesteps the race entirely.
    4. If the batch is rejected with 13056 — or the entry is a limit that
       won't fill synchronously — fall through: place only the entry, then
       spawn a background task that awaits the ws ``order.filled`` /
       ``position.opened`` event before attaching SL/TP. The background
       task owns the existing rollback sequence on SL failure.

    Returns ``(ok, summary_message)``. The summary is intended for direct
    inclusion in the user-facing confirmation. The background task (when
    spawned) emits its own follow-up once brackets attach.
    """
    # Step 1: leverage.
    if leverage is not None and leverage > 0:
        try:
            await _maybe_set_leverage(client, account_id, asset, leverage)
        except Exception as exc:  # noqa: BLE001
            return False, f"leverage step failed: {exc}"

    direction_upper = "LONG" if side == "long" else "SHORT"
    taker_side = "buy" if side == "long" else "sell"
    opposite_side = "sell" if taker_side == "buy" else "buy"
    entry_desc = (
        f"limit {fmt_num(entry_price)}"
        if order_type == "limit" and entry_price is not None
        else "market"
    )
    has_brackets = (
        stop_loss is not None and stop_loss > 0
    ) or bool(take_profits)

    # Step 2: no brackets requested → plain entry, no race to avoid.
    if not has_brackets:
        try:
            await client.place_order(
                account_id=account_id,
                asset=asset,
                type=order_type,
                side=taker_side,
                positionSide=side,
                timeInForce="GTC" if order_type == "limit" else "IOC",
                quantity=quantity,
                price=entry_price if order_type == "limit" else None,
                reduceOnly=False,
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"entry placement failed: {exc}"
        summary = _format_summary(
            direction_upper=direction_upper,
            quantity=quantity,
            asset=asset,
            entry_desc=entry_desc,
            stop_loss=None,
            take_profits=[],
        )
        return True, summary

    # Step 3: atomic batch attempt. Market entries are the happy path for
    # this branch — the server typically groups all legs immediately.
    # review finding 13056: single POST lets Propr attach conditionals to
    # the resulting position atomically, dodging the race entirely.
    legs: List[dict] = [
        _build_entry_leg(
            account_id=account_id,
            asset=asset,
            side=side,
            taker_side=taker_side,
            order_type=order_type,
            quantity=quantity,
            entry_price=entry_price,
        )
    ]
    if stop_loss is not None and stop_loss > 0:
        legs.append(
            _build_sl_leg(
                account_id=account_id,
                asset=asset,
                side=side,
                opposite_side=opposite_side,
                quantity=quantity,
                stop_loss=stop_loss,
            )
        )
    if take_profits:
        qtys = _tp_qtys(quantity, len(take_profits))
        for leg_qty, tp in zip(qtys, take_profits):
            legs.append(
                _build_tp_leg(
                    account_id=account_id,
                    asset=asset,
                    side=side,
                    opposite_side=opposite_side,
                    quantity=leg_qty,
                    trigger=tp,
                )
            )

    # Limit entries almost always need the deferred path (the position
    # doesn't exist yet), so skip the batch attempt for them to avoid a
    # guaranteed 13056 round-trip.
    try_batch = order_type != "limit"
    batch_ok = False
    if try_batch:
        try:
            await client.place_batch(account_id, legs)
            batch_ok = True
        except ProprAPIError as exc:
            if is_conditional_requires_position(exc):
                # review finding 13056: fall through to deferred brackets.
                log.info(
                    "batch rejected with 13056; deferring brackets for %s %s",
                    side,
                    asset,
                )
            else:
                return False, f"batch entry failed: {exc}"
        except Exception as exc:  # noqa: BLE001
            return False, f"batch entry failed: {exc}"

    if batch_ok:
        summary = _format_summary(
            direction_upper=direction_upper,
            quantity=quantity,
            asset=asset,
            entry_desc=entry_desc,
            stop_loss=stop_loss,
            take_profits=take_profits,
        )
        return True, f"Order placed (atomic): {summary}"

    # Step 4: deferred fallback — place entry only, then attach brackets
    # after the ``order.filled`` / ``position.opened`` event arrives.
    try:
        entry_response = await client.place_order(
            account_id=account_id,
            asset=asset,
            type=order_type,
            side=taker_side,
            positionSide=side,
            timeInForce="GTC" if order_type == "limit" else "IOC",
            quantity=quantity,
            price=entry_price if order_type == "limit" else None,
            reduceOnly=False,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"entry placement failed: {exc}"

    entry_order_id = _extract_order_id(entry_response)
    task = asyncio.create_task(
        _place_brackets_after_fill(
            bot=bot,
            chat_id=chat_id,
            client=client,
            account_id=account_id,
            asset=asset,
            side=side,
            order_type=order_type,
            entry_order_id=entry_order_id,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profits=list(take_profits),
        ),
        name=f"brackets-{asset}-{entry_order_id or 'noid'}",
    )
    _track_bracket_task(bot_data, task)

    entry_summary = (
        f"{direction_upper} {fmt_num(quantity, 6)} {asset} @ {entry_desc}"
    )
    deferred = (
        f"Entry placed: {entry_summary}. Brackets will attach when it fills."
    )
    return True, deferred


# ---------------------------------------------------------------------------
# Natural-language trade intents — /trade + free-form text handler
# ---------------------------------------------------------------------------
_INTENT_REQUIRED_FIELDS: dict[str, Tuple[str, ...]] = {
    "open": ("asset", "side"),                 # quantity/usd_amount validated separately
    "close": ("asset",),
    "sl": ("asset", "stop_loss"),
    "tp": ("asset", "take_profit"),
    "setleverage": ("asset", "leverage"),
    "analysis": ("asset",),
}


def _intent_decimal(value: Any) -> Optional[Decimal]:
    """Coerce an intent JSON string/number to ``Decimal``; ``None`` on failure."""
    if value is None or value == "":
        return None
    try:
        d = Decimal(str(value).replace(",", "").replace("_", ""))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return d


def _intent_int(value: Any) -> Optional[int]:
    """Coerce an intent JSON value to ``int``; ``None`` on failure."""
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except (ValueError, TypeError):
        return None


def _quantize_qty(qty: Decimal, price: Decimal) -> Decimal:
    """Round a computed quantity to a sensible precision for the asset price.

    Cheap assets (price < $10, e.g. DOGE / PEPE) get 6 dp; everything else 4 dp.
    """
    step = Decimal("0.000001") if price < Decimal(10) else Decimal("0.0001")
    return qty.quantize(step)


def _intent_missing_fields(intent: dict) -> List[str]:
    """Return the list of required fields missing for *intent*."""
    kind = str(intent.get("intent", "")).lower()
    required = _INTENT_REQUIRED_FIELDS.get(kind, ())
    missing = [f for f in required if intent.get(f) in (None, "")]
    if kind == "open":
        qty = _intent_decimal(intent.get("quantity"))
        usd = _intent_decimal(intent.get("usd_amount"))
        if qty is None and usd is None:
            missing.append("quantity or usd_amount")
    return missing


def _render_intent_card(
    intent: dict,
    resolved_qty: Optional[Decimal],
    price_at_parse: Optional[Decimal],
    extra_notes: List[str],
) -> str:
    """Build the MarkdownV2 confirmation card shown before execution."""
    kind = str(intent.get("intent", "")).lower()
    asset = str(intent.get("asset", "?"))
    side = str(intent.get("side") or "").lower()
    leverage = _intent_int(intent.get("leverage"))
    order_type = str(intent.get("order_type") or "market").lower()
    notes = str(intent.get("notes") or "")

    header = "🧠 *Parsed intent*"
    lines: List[str] = [header]

    if kind == "open":
        action = f"OPEN {side.upper() or '?'}"
        lines.append(f"Action: {escape_md(action)}")
        lines.append(f"Asset: {escape_md(asset)}")

        qty_str = fmt_num(resolved_qty, 6) if resolved_qty is not None else "?"
        usd_amount = _intent_decimal(intent.get("usd_amount"))
        lev_label = f"{leverage}x" if leverage else "1x"
        if usd_amount is not None and resolved_qty is not None:
            # review finding 3: disambiguate USD-sized opens so users see
            # margin vs notional and the leverage multiplier explicitly.
            lev_mul = Decimal(leverage) if leverage else Decimal(1)
            notional = usd_amount * lev_mul
            lines.append(
                f"Size: {escape_md(qty_str)} {escape_md(asset)} "
                f"\\(\\~${escape_md(fmt_num(notional))} notional, "
                f"${escape_md(fmt_num(usd_amount))} margin @ {escape_md(lev_label)}\\)"
            )
        else:
            lines.append(f"Size: {escape_md(qty_str)} {escape_md(asset)}")

        if order_type == "limit":
            limit_price = _intent_decimal(intent.get("limit_price"))
            price_desc = f"limit @ {fmt_num(limit_price)}" if limit_price else "limit"
            lines.append(f"Entry: {escape_md(price_desc)}")
        else:
            lines.append("Entry: market")

        if leverage:
            lines.append(f"Leverage: {escape_md(lev_label)}")

        sl = _intent_decimal(intent.get("stop_loss"))
        tp = _intent_decimal(intent.get("take_profit"))
        if sl is not None:
            lines.append(f"SL: {escape_md(fmt_num(sl))}")
        if tp is not None:
            lines.append(f"TP: {escape_md(fmt_num(tp))}")

    elif kind == "close":
        side_label = side.upper() if side else "AUTO"
        lines.append(f"Action: CLOSE {escape_md(side_label)}")
        lines.append(f"Asset: {escape_md(asset)}")

    elif kind == "sl":
        sl = _intent_decimal(intent.get("stop_loss"))
        lines.append("Action: SET STOP\\-LOSS")
        lines.append(f"Asset: {escape_md(asset)}")
        lines.append(f"Trigger: {escape_md(fmt_num(sl))}")

    elif kind == "tp":
        tp = _intent_decimal(intent.get("take_profit"))
        lines.append("Action: SET TAKE\\-PROFIT")
        lines.append(f"Asset: {escape_md(asset)}")
        lines.append(f"Trigger: {escape_md(fmt_num(tp))}")

    elif kind == "setleverage":
        lev_label = f"{leverage}x" if leverage else "?"
        lines.append("Action: SET LEVERAGE")
        lines.append(f"Asset: {escape_md(asset)}")
        lines.append(f"Leverage: {escape_md(lev_label)}")

    elif kind == "analysis":
        timeframe = str(intent.get("timeframe") or "1h")
        lines.append("Action: ANALYSIS")
        lines.append(f"Asset: {escape_md(asset)}")
        lines.append(f"Timeframe: {escape_md(timeframe)}")

    lines.append("─" * 16)
    if notes:
        lines.append(f'"{escape_md(notes)}"')
    for extra in extra_notes:
        lines.append(f"_{escape_md(extra)}_")
    return "\n".join(lines)


@authorized
async def cmd_trade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/trade <natural language prompt>``

    Same behavior as sending the prompt as a plain message — useful when the
    first word looks like a reserved command (e.g. ``close`` is not a slash
    command but we want it to route as a close intent).
    """
    try:
        text = " ".join(context.args or []).strip()
        if not text:
            await _reply(
                update,
                'Usage: `/trade <prompt>` \\(e\\.g\\. `/trade open long on btc with $4000`\\)',
            )
            return
        await _process_prompt(update, context, text)
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any non-command text message as a natural-language trade intent."""
    try:
        message = update.effective_message
        if message is None:
            return
        text = (message.text or "").strip()
        if not text:
            return
        if text.startswith("/"):
            # The MessageHandler filter already excludes commands; this is just
            # a safety net for edge cases (e.g. forwarded messages with a
            # leading slash).
            return
        await _process_prompt(update, context, text)
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


async def _process_prompt(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
) -> None:
    """Shared NL core: Groq parse → resolve qty → show confirmation card."""
    chat_id = update.effective_chat.id

    # Step 1: typing indicator + "thinking" placeholder so the user sees
    # something while Groq is running.
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    placeholder = await context.bot.send_message(
        chat_id=chat_id, text="✍️ Thinking..."
    )

    async def _typing_loop() -> None:
        try:
            while True:
                await asyncio.sleep(4)
                await context.bot.send_chat_action(
                    chat_id=chat_id, action=ChatAction.TYPING
                )
        except asyncio.CancelledError:
            return

    typing_task = asyncio.create_task(_typing_loop())
    try:
        intent = await parse_trade_intent(text)
    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    # Drop the placeholder now that we have a result.
    try:
        await context.bot.delete_message(
            chat_id=chat_id, message_id=placeholder.message_id
        )
    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
        log.debug("delete placeholder failed: %s", exc)

    kind = str(intent.get("intent", "")).lower()
    notes_text = str(intent.get("notes") or "").strip()

    if kind == "unknown" or kind == "":
        # review finding 15: always surface the parser's notes; if empty fall
        # back to a generic human-readable reason so the reply never reads
        # like an unexplained rejection.
        reason = notes_text or "The request didn't map to a supported action."
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❓ Sorry, I couldn't parse that\\. {escape_md(reason)}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    # Normalize / upper-case asset (parser already does this, defensive copy).
    if isinstance(intent.get("asset"), str):
        intent["asset"] = intent["asset"].split(":")[-1].upper()

    missing = _intent_missing_fields(intent)
    if missing:
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"⚠️ Missing fields for {escape_md(kind)}: "
                f"`{escape_md(', '.join(missing))}`"
            ),
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    extra_notes: List[str] = []
    resolved_qty: Optional[Decimal] = None
    price_at_parse: Optional[Decimal] = None

    # Step 2: open-intent qty resolution + leverage clamp.
    if kind == "open":
        asset = str(intent["asset"])
        qty_raw = _intent_decimal(intent.get("quantity"))
        usd_raw = _intent_decimal(intent.get("usd_amount"))
        leverage = _intent_int(intent.get("leverage"))

        if leverage is not None:
            cap = max_leverage_for(asset)
            if leverage > cap:
                extra_notes.append(f"Leverage clamped to {cap}x (max for {asset}).")
                leverage = cap
                intent["leverage"] = cap
            elif leverage <= 0:
                extra_notes.append("Leverage ≤ 0 ignored; using 1x.")
                leverage = None
                intent["leverage"] = None

        if qty_raw is not None and qty_raw > 0:
            resolved_qty = qty_raw
        elif usd_raw is not None and usd_raw > 0:
            price = await fetch_spot_price(asset)
            if price is None or price <= 0:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"⚠️ Could not fetch {escape_md(asset)} price — try again "
                        "or specify quantity directly\\."
                    ),
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                return
            lev_mul = Decimal(leverage) if leverage else Decimal(1)
            try:
                # review finding 3: ``qty = (usd × leverage) / price`` is
                # intentional — the usd_amount is the MARGIN the user wants
                # to deploy, so the resulting notional is ``usd × leverage``
                # (e.g. $4000 margin at 5x → $20,000 notional → 0.267 BTC
                # @ $75k). Card text below spells this out unambiguously.
                raw_qty = (usd_raw * lev_mul) / price
            except (InvalidOperation, ZeroDivisionError) as exc:
                log.warning("qty compute failed: %s", exc)
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"⚠️ Could not compute quantity: {escape_md(exc)}",
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                return
            resolved_qty = _quantize_qty(raw_qty, price)
            price_at_parse = price
            if resolved_qty <= 0:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "⚠️ Computed quantity rounded to zero — try a larger "
                        "USD amount or specify the size directly\\."
                    ),
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                return
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="⚠️ Need a quantity or USD amount for open intents\\.",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
            return

    # Step 3: setleverage clamp (independent of open).
    if kind == "setleverage":
        asset = str(intent["asset"])
        leverage = _intent_int(intent.get("leverage"))
        if leverage is None or leverage <= 0:
            await context.bot.send_message(
                chat_id=chat_id,
                text="⚠️ Leverage must be a positive integer\\.",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
            return
        cap = max_leverage_for(asset)
        if leverage > cap:
            extra_notes.append(f"Leverage clamped to {cap}x (max for {asset}).")
            leverage = cap
            intent["leverage"] = cap

    # Step 4: analysis timeframe default.
    if kind == "analysis":
        tf = intent.get("timeframe")
        if not tf or tf not in VALID_INTERVALS:
            if tf:
                extra_notes.append(f"Unknown timeframe {tf!r}; defaulting to 1h.")
            intent["timeframe"] = "1h"

    # Step 5: render confirmation card + keyboard.
    pending_id = str(ULID())
    card = _render_intent_card(intent, resolved_qty, price_at_parse, extra_notes)

    keyboard = InlineKeyboardMarkup(
        [[
            InlineKeyboardButton(
                "✅ Execute", callback_data=f"exec_intent:{pending_id}"
            ),
            InlineKeyboardButton(
                "❌ Cancel", callback_data=f"skip_intent:{pending_id}"
            ),
        ]]
    )

    sent = await context.bot.send_message(
        chat_id=chat_id,
        text=card,
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=keyboard,
    )

    PENDING_INTENTS[pending_id] = PendingIntent(
        pending_id=pending_id,
        chat_id=chat_id,
        intent=intent,
        resolved_qty=resolved_qty,
        price_at_parse=price_at_parse,
        created_at=datetime.now(timezone.utc),
        message_id=sent.message_id,
        notes=list(extra_notes),
        # review finding 14: persist the original card markdown so the
        # sweeper can edit it with "⏰ Confirmation expired" on TTL.
        message_text=card,
    )


# ---------------------------------------------------------------------------
# Intent callback dispatch
# ---------------------------------------------------------------------------
async def _handle_intent_callback(
    context: ContextTypes.DEFAULT_TYPE,
    query: Any,
    chat_id: int,
    action: str,
    pending_id: str,
) -> None:
    """Route an ``exec_intent`` / ``skip_intent`` callback to execution."""
    pending = PENDING_INTENTS.get(pending_id)
    if pending is None:
        await query.answer("Expired")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception as exc:  # noqa: BLE001
            log.debug("edit_message_reply_markup on expired intent failed: %s", exc)
        return

    if action == "skip_intent":
        PENDING_INTENTS.pop(pending_id, None)
        await query.answer("Cancelled")
        try:
            original_md = query.message.text_markdown_v2 or ""
            await query.edit_message_text(
                text=original_md + "\n\n⏭ Skipped",
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=None,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("skip intent cleanup failed: %s", exc)
        return

    # action == "exec_intent"
    await query.answer("Executing…")

    PENDING_INTENTS.pop(pending_id, None)

    # review finding 10: analysis intents previously reported "✅ Executed"
    # immediately after queueing the nested /analysis flow — the flow was
    # fire-and-forget. Now we update the card to "Running analysis…" first,
    # await the real pipeline, and only then report success / failure.
    kind = str(pending.intent.get("intent", "")).lower()
    if kind == "analysis":
        asset = str(pending.intent.get("asset") or "").upper()
        timeframe = str(pending.intent.get("timeframe") or "1h")
        await _append_status_to_intent_message(
            query, f"✍️ Running analysis on {asset}…"
        )
        try:
            await _run_analysis_flow(
                context,
                chat_id=chat_id,
                asset=asset,
                timeframe=timeframe,
                direction=pending.intent.get("side"),
                intro_reply=None,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("analysis intent failed: %s", exc)
            await _append_status_to_intent_message(
                query, f"❌ Failed: {exc}"
            )
            return
        await _append_status_to_intent_message(query, "✅ Done")
        return

    try:
        summary = await _execute_intent(context, chat_id, pending)
    except ProprAPIError as exc:
        await _append_status_to_intent_message(query, f"❌ {exc}")
        return
    except Exception as exc:  # noqa: BLE001 — never crash the callback handler
        log.exception("intent execution failed: %s", exc)
        await _append_status_to_intent_message(
            query, f"❌ Unexpected error: {type(exc).__name__}: {exc}"
        )
        return

    await _append_status_to_intent_message(query, f"✅ Executed\n{summary}")


async def _append_status_to_intent_message(query: Any, status: str) -> None:
    """Append a status line (already plain text) to the intent confirmation card."""
    try:
        original_md = query.message.text_markdown_v2 or ""
        new_text = (original_md + "\n\n" + escape_md(status)).strip()
        await query.edit_message_text(
            text=new_text,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=None,
        )
    except Exception as exc:  # noqa: BLE001 — fall back to a fresh message
        log.debug("edit status onto intent message failed: %s", exc)
        try:
            await query.message.reply_text(status)
        except Exception as reply_exc:  # noqa: BLE001
            log.debug("fallback status send failed: %s", reply_exc)


async def _execute_intent(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    pending: "PendingIntent",
) -> str:
    """Dispatch a confirmed intent to the right Propr/analysis call.

    Returns a short human-readable summary of what was done. Raises
    :class:`ProprAPIError` (or any other exception) on failure — the caller
    translates to a friendly user message.
    """
    client = _get_client(context)
    account_id = await _ensure_account_id(context)

    intent = pending.intent
    kind = str(intent.get("intent", "")).lower()
    asset = str(intent.get("asset") or "").upper()

    if kind == "open":
        return await _execute_intent_open(
            client, account_id, context, chat_id, pending
        )
    if kind == "close":
        return await _execute_intent_close(client, account_id, intent)
    if kind == "sl":
        return await _execute_intent_sl(client, account_id, intent)
    if kind == "tp":
        return await _execute_intent_tp(client, account_id, intent)
    if kind == "setleverage":
        return await _execute_intent_set_leverage(client, account_id, intent)
    # review finding 10: the analysis branch is handled directly in
    # _handle_intent_callback so the card can reflect in-progress / done
    # state against the real flow rather than declaring success on queue.
    raise ProprAPIError(400, f"Unsupported intent: {kind}")


async def _execute_intent_open(
    client: ProprClient,
    account_id: str,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    pending: "PendingIntent",
) -> str:
    """Execute an ``open`` intent: optional leverage + entry + optional SL/TP.

    Delegates to :func:`_execute_open_with_brackets` so the entry+SL+TP
    placement goes through the atomic-batch-with-deferred-fallback path
    that dodges the 13056
    ``conditional_order_requires_position_or_group`` race.
    """
    intent = pending.intent
    asset = str(intent.get("asset") or "")
    side = str(intent.get("side") or "").lower()
    if side not in ("long", "short"):
        raise ProprAPIError(400, "open intent requires side=long|short")
    leverage = _intent_int(intent.get("leverage"))
    order_type = str(intent.get("order_type") or "market").lower()
    limit_price = _intent_decimal(intent.get("limit_price"))
    stop_loss = _intent_decimal(intent.get("stop_loss"))
    take_profit = _intent_decimal(intent.get("take_profit"))

    quantity = pending.resolved_qty
    if quantity is None:
        quantity = _intent_decimal(intent.get("quantity"))
    if quantity is None or quantity <= 0:
        raise ProprAPIError(400, "open intent has no usable quantity")

    take_profits: List[Decimal] = (
        [take_profit] if take_profit is not None and take_profit > 0 else []
    )

    ok, summary = await _execute_open_with_brackets(
        bot=context.bot,
        chat_id=chat_id,
        bot_data=context.application.bot_data,
        client=client,
        account_id=account_id,
        asset=asset,
        side=side,
        order_type=order_type,
        quantity=quantity,
        entry_price=limit_price if order_type == "limit" else None,
        stop_loss=stop_loss,
        take_profits=take_profits,
        leverage=leverage,
    )
    if not ok:
        # Surface as ProprAPIError so the NL callback's status line renders
        # the failure in the same shape as every other intent kind.
        raise ProprAPIError(0, summary)
    return summary


async def _execute_intent_close(
    client: ProprClient,
    account_id: str,
    intent: dict,
) -> str:
    """Execute a ``close`` intent: optional side, auto-detect if missing."""
    asset = str(intent.get("asset") or "")
    # review finding 4: compare via venue ticker so alias asset (OIL) matches
    # the xyz:CL position row that the API returns.
    asset_norm_ticker = asset_ticker(normalize_asset(asset))
    side = str(intent.get("side") or "").lower()

    if side not in ("long", "short"):
        positions = await client.get_positions(account_id)
        target = next(
            (p for p in positions if _pos_asset_upper(p) == asset_norm_ticker),
            None,
        )
        if target is None:
            raise ProprAPIError(404, f"No open position on {asset}")
        side = _pos_side(target)
        if side not in ("long", "short"):
            raise ProprAPIError(404, f"Could not determine position side for {asset}")

    await client.close_position(account_id, asset, side)
    return f"Closed {side.upper()} {asset}"


async def _execute_intent_sl(
    client: ProprClient,
    account_id: str,
    intent: dict,
) -> str:
    """Execute an ``sl`` intent: attach stop-loss to the current position."""
    asset = str(intent.get("asset") or "")
    # review finding 4: same venue-ticker lookup as _execute_intent_close.
    asset_norm_ticker = asset_ticker(normalize_asset(asset))
    trigger = _intent_decimal(intent.get("stop_loss"))
    if trigger is None or trigger <= 0:
        raise ProprAPIError(400, "sl intent requires a positive stop_loss")
    side_hint = str(intent.get("side") or "").lower()

    positions = await client.get_positions(account_id)
    candidates = [p for p in positions if _pos_asset_upper(p) == asset_norm_ticker]
    if side_hint in ("long", "short"):
        candidates = [p for p in candidates if _pos_side(p) == side_hint]
    target = candidates[0] if candidates else None
    if target is None:
        raise ProprAPIError(404, f"No open position on {asset}")

    side = _pos_side(target)
    if side not in ("long", "short"):
        raise ProprAPIError(404, f"Could not determine position side for {asset}")
    qty = target.get("quantity", target.get("qty"))
    await client.stop_loss(account_id, asset, qty, trigger, side)
    return f"SL set on {side.upper()} {asset} @ {fmt_num(trigger)}"


async def _execute_intent_tp(
    client: ProprClient,
    account_id: str,
    intent: dict,
) -> str:
    """Execute a ``tp`` intent: attach take-profit to the current position."""
    asset = str(intent.get("asset") or "")
    # review finding 4: resolve alias to venue ticker for position compare.
    asset_norm_ticker = asset_ticker(normalize_asset(asset))
    trigger = _intent_decimal(intent.get("take_profit"))
    if trigger is None or trigger <= 0:
        raise ProprAPIError(400, "tp intent requires a positive take_profit")
    side_hint = str(intent.get("side") or "").lower()

    positions = await client.get_positions(account_id)
    candidates = [p for p in positions if _pos_asset_upper(p) == asset_norm_ticker]
    if side_hint in ("long", "short"):
        candidates = [p for p in candidates if _pos_side(p) == side_hint]
    target = candidates[0] if candidates else None
    if target is None:
        raise ProprAPIError(404, f"No open position on {asset}")

    side = _pos_side(target)
    if side not in ("long", "short"):
        raise ProprAPIError(404, f"Could not determine position side for {asset}")
    qty = target.get("quantity", target.get("qty"))
    await client.take_profit(account_id, asset, qty, trigger, side)
    return f"TP set on {side.upper()} {asset} @ {fmt_num(trigger)}"


async def _execute_intent_set_leverage(
    client: ProprClient,
    account_id: str,
    intent: dict,
) -> str:
    """Execute a ``setleverage`` intent."""
    asset = str(intent.get("asset") or "")
    leverage = _intent_int(intent.get("leverage"))
    if leverage is None or leverage <= 0:
        raise ProprAPIError(400, "setleverage intent requires a positive leverage")
    await client.set_leverage(account_id, asset, leverage)
    return f"Leverage on {asset} set to {leverage}x"


# ---------------------------------------------------------------------------
# PnL commands
# ---------------------------------------------------------------------------
@authorized
async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/pnl`` — text snapshot of realized + unrealized PnL.

    ``/pnl image`` returns a Propr-style share card PNG rendered via Pillow
    (focusing on the position with the largest absolute uPnL). Empty
    positions still produce a neutral "No open positions" card.
    """
    try:
        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        raw_positions = await client.get_positions(account_id)
        snapshot = build_snapshot(raw_positions)

        args = [a.lower() for a in (context.args or [])]
        if "image" in args:
            chat_id = update.effective_chat.id
            total = snapshot.total
            sign = "+" if total >= 0 else ""
            caption = f"✅ Total PnL: {sign}{fmt_num(total)} USDC"
            try:
                png_bytes = await asyncio.to_thread(render_snapshot_image, snapshot)
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=InputFile(io.BytesIO(png_bytes), filename="pnl.png"),
                    caption=caption,
                )
            except Exception as render_exc:  # noqa: BLE001 — Pillow / font failures
                log.error(
                    "render_snapshot_image failed: %s\n%s",
                    render_exc,
                    traceback.format_exc(),
                )
                await _reply(update, format_snapshot_markdown(snapshot, escape_md))
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="⚠️ Could not render PnL card — showing text instead.",
                )
            return

        await _reply(update, format_snapshot_markdown(snapshot, escape_md))
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def cmd_livepnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/livepnl`` — toggle live PnL push notifications on position updates.

    ``/livepnl`` alone reports the current state; ``/livepnl on|off``
    enables/disables the websocket-driven live pushes.
    """
    try:
        bot_data = context.application.bot_data
        args = [a.lower() for a in (context.args or [])]

        if args:
            choice = args[0]
            if choice in ("on", "enable", "enabled", "true", "1"):
                bot_data["live_pnl_enabled"] = True
            elif choice in ("off", "disable", "disabled", "false", "0"):
                bot_data["live_pnl_enabled"] = False
            else:
                await _reply(update, "Usage: `/livepnl [on|off]`")
                return

        enabled = bool(bot_data.get("live_pnl_enabled"))
        label = "🔴 Live PnL: ON" if enabled else "⚪ Live PnL: OFF"
        await _reply(update, escape_md(label))
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/help`` — list every command with a short example."""
    text = (
        "*Propr Bot Commands*\n\n"
        "• `/start` — welcome \\+ health check\n"
        "• `/status` — API \\+ challenge \\+ P&L\n"
        "• `/positions` — open positions\n"
        "• `/orders` — open orders\n"
        "• `/trades [n]` — last n trades \\(default 5\\)\n"
        "• `/buy <asset> <qty> [price]` — e\\.g\\. `/buy BTC 0.001`\n"
        "• `/sell <asset> <qty> [price]` — reduce\\-only\n"
        "• `/close <asset>` — close a position\n"
        "• `/cancel <orderId>` — cancel one order\n"
        "• `/cancelall` — cancel every open order\n"
        "• `/sl <asset> <trigger>` — stop\\-loss on current position \\(long or short\\)\n"
        "• `/tp <asset> <trigger>` — take\\-profit on current position \\(long or short\\)\n"
        "• `/leverage <asset> <n>` — set leverage\n"
        "• `/analysis <asset> [tf] [dir]` — AI trade plan\n"
        "• `/pnl [image]` — show realized \\+ unrealized PnL \\(append `image` for PNG card\\)\n"
        "• `/livepnl [on|off]` — toggle real\\-time PnL updates\n"
        "• Natural language — just type what you want, "
        'e\\.g\\. _"open long on btc with $4000 and 5x leverage"_\n'
        "• `/trade <prompt>` — same as above but explicit\n"
        "• `/help` — this message"
    )
    await _reply(update, text)


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log uncaught errors and notify the authorized chat."""
    log.exception("Unhandled error: %s", context.error)
    raw = os.getenv("TELEGRAM_CHAT_ID", "")
    try:
        chat_id = int(raw)
    except ValueError:
        return
    try:
        await context.bot.send_message(
            chat_id=chat_id, text="❌ Internal error — check logs"
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to notify chat of error: %s", exc)


# ---------------------------------------------------------------------------
# Handler registry (consumed by bot.py)
# ---------------------------------------------------------------------------
COMMAND_HANDLERS: List[Tuple[str, HandlerFn]] = [
    ("start", start_cmd),
    ("status", status_cmd),
    ("positions", positions_cmd),
    ("orders", orders_cmd),
    ("trades", trades_cmd),
    ("buy", buy_cmd),
    ("sell", sell_cmd),
    ("close", close_cmd),
    ("cancel", cancel_cmd),
    ("cancelall", cancelall_cmd),
    ("sl", sl_cmd),
    ("tp", tp_cmd),
    ("leverage", leverage_cmd),
    ("analysis", analysis_cmd),
    ("trade", cmd_trade),
    ("pnl", cmd_pnl),
    ("livepnl", cmd_livepnl),
    ("help", help_cmd),
]
