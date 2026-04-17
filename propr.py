"""Async REST client for the Propr prop-firm trading API.

All calls use ``httpx.AsyncClient`` with a 30 second timeout. Non-2xx responses
raise :class:`ProprAPIError` which Telegram handlers render as friendly
messages. Order helpers always generate a fresh ULID ``intentId`` internally.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

import httpx
from ulid import ULID

log = logging.getLogger(__name__)

BASE_URL = "https://api.propr.xyz/v1"

# Assets on Hyperliquid's HIP-3 rail (equities / commodities) need a ``xyz:``
# prefix; crypto perps (BTC, ETH, ...) pass through as-is. Ticker *output*
# may differ from the user-facing alias — e.g. "OIL" routes to the
# ``xyz:CL`` WTI-crude perp (PR #1 review finding #8).
HIP3_ASSETS: Dict[str, str] = {
    "AAPL": "AAPL", "TSLA": "TSLA", "NVDA": "NVDA", "MSFT": "MSFT",
    "META": "META", "GOOGL": "GOOGL", "AMZN": "AMZN",
    "GOLD": "GOLD", "SILVER": "SILVER",
    "OIL": "CL", "CL": "CL",
}

# Per-asset leverage caps enforced before any margin-config PUT.
MAX_LEVERAGE_BTC_ETH = 5
MAX_LEVERAGE_OTHER = 2


class ProprAPIError(Exception):
    """Raised when the Propr API returns a non-success status."""

    def __init__(self, status: int, body: Any):
        self.status = status
        self.body = body
        super().__init__(f"Propr API error {status}: {body}")


def normalize_asset(asset: str) -> str:
    """Upper-case an asset symbol and prefix HIP-3 assets with ``xyz:``.

    Crypto assets (BTC, ETH, SOL, ...) are returned unchanged. HIP-3 assets
    (AAPL, TSLA, GOLD, ...) are prefixed so the API routes to the correct
    venue. Already-prefixed values are returned unchanged.
    """
    if not asset:
        return asset
    if ":" in asset:
        return asset
    upper = asset.upper()
    if upper in HIP3_ASSETS:
        # Use the mapped venue ticker so aliases (OIL → CL) route correctly.
        return f"xyz:{HIP3_ASSETS[upper]}"
    return upper


def asset_ticker(asset: str) -> str:
    """Strip any exchange prefix like ``xyz:`` and upper-case.

    Used for comparing user-facing tickers against position rows where the
    API echoes back the normalised ``xyz:CL`` form (review finding 4).
    """
    # review finding 4: single source of truth for ticker comparisons so
    # OIL→CL aliasing doesn't silently break downstream lookups.
    if not asset:
        return ""
    return asset.split(":")[-1].upper()


def max_leverage_for(asset: str) -> int:
    """Return the enforced maximum leverage for *asset*."""
    base = asset_ticker(asset)
    if base in ("BTC", "ETH"):
        return MAX_LEVERAGE_BTC_ETH
    return MAX_LEVERAGE_OTHER


class ProprClient:
    """Thin async wrapper around the Propr REST API."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("api_key must be a non-empty string")
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=30.0,
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        self._active_account_id: Optional[str] = None

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Core request helper
    # ------------------------------------------------------------------
    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        allow_status: Optional[tuple] = None,
    ) -> Any:
        """Send a request and return the parsed JSON body.

        Accepts both ``200`` and ``201`` as success. Any other status raises
        :class:`ProprAPIError`.
        """
        try:
            resp = await self._client.request(method, path, params=params, json=json)
        except httpx.HTTPError as exc:
            log.error("Propr %s %s network error: %s", method, path, exc)
            raise ProprAPIError(0, str(exc)) from exc

        log.debug("Propr %s %s -> %s", method, path, resp.status_code)

        if resp.status_code in (200, 201):
            if not resp.content:
                return {}
            try:
                return resp.json()
            except ValueError:
                return resp.text

        if allow_status and resp.status_code in allow_status:
            try:
                return resp.json()
            except ValueError:
                return {"status": resp.status_code, "body": resp.text}

        # Try to surface a useful body without blowing up on non-JSON errors.
        try:
            body: Any = resp.json()
        except ValueError:
            body = resp.text
        raise ProprAPIError(resp.status_code, body)

    # ------------------------------------------------------------------
    # Simple reads
    # ------------------------------------------------------------------
    async def health(self) -> Dict[str, Any]:
        """GET /health — liveness check."""
        return await self._request("GET", "/health")

    async def me(self) -> Dict[str, Any]:
        """GET /users/me — returns the authenticated user object."""
        return await self._request("GET", "/users/me")

    async def get_challenge_attempts(self) -> List[Dict[str, Any]]:
        """GET /challenge-attempts — list all challenge attempts."""
        data = await self._request("GET", "/challenge-attempts")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "attempts", "items", "results"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        return []

    async def get_active_account_id(self) -> str:
        """Return the ``accountId`` of the first active challenge attempt.

        Result is cached after the first successful lookup so that repeated
        handler calls don't refetch on every command.
        """
        if self._active_account_id:
            return self._active_account_id
        attempts = await self.get_challenge_attempts()
        if not attempts:
            raise ProprAPIError(404, "No challenge attempts found for this API key")

        def _is_active(a: Dict[str, Any]) -> bool:
            status = str(a.get("status", "")).lower()
            return status in ("", "active", "in_progress", "running", "open")

        active = next((a for a in attempts if _is_active(a)), None) or attempts[0]
        account_id = (
            active.get("accountId")
            or active.get("account_id")
            or (active.get("account") or {}).get("id")
        )
        if not account_id:
            raise ProprAPIError(404, "Active challenge attempt has no accountId")
        self._active_account_id = str(account_id)
        log.info("Cached active accountId=%s", self._active_account_id)
        return self._active_account_id

    async def get_positions(self, account_id: str) -> List[Dict[str, Any]]:
        """GET /accounts/{id}/positions filtered to non-zero quantity."""
        data = await self._request("GET", f"/accounts/{account_id}/positions")
        rows = _as_list(data)

        def _qty(row: Dict[str, Any]) -> str:
            return str(row.get("quantity", row.get("qty", "0")))

        return [r for r in rows if _qty(r) not in ("0", "0.0", "", "0.00")]

    async def get_open_orders(self, account_id: str) -> List[Dict[str, Any]]:
        """GET /accounts/{id}/orders?status=open."""
        data = await self._request(
            "GET", f"/accounts/{account_id}/orders", params={"status": "open"}
        )
        return _as_list(data)

    async def get_trades(self, account_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """GET /accounts/{id}/trades — most recent *limit* trades."""
        data = await self._request(
            "GET", f"/accounts/{account_id}/trades", params={"limit": limit}
        )
        return _as_list(data)[:limit]

    async def get_leverage_limits(self) -> Dict[str, Any]:
        """GET /leverage-limits/effective — user-specific leverage caps."""
        return await self._request("GET", "/leverage-limits/effective")

    async def get_margin_config(self, account_id: str, asset: str) -> Dict[str, Any]:
        """GET /accounts/{id}/margin-config/{asset}."""
        asset_norm = normalize_asset(asset)
        return await self._request(
            "GET", f"/accounts/{account_id}/margin-config/{asset_norm}"
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    async def set_leverage(
        self, account_id: str, asset: str, leverage: int
    ) -> Dict[str, Any]:
        """Update the leverage on the margin config for *asset*.

        Enforces per-asset caps (BTC/ETH max 5x, everything else max 2x).
        Uses the existing margin config to find the ``configId`` required by
        the PUT endpoint.
        """
        if leverage <= 0:
            raise ValueError("leverage must be a positive integer")
        cap = max_leverage_for(asset)
        if leverage > cap:
            raise ProprAPIError(
                400,
                f"Leverage {leverage}x exceeds max {cap}x for {asset}",
            )
        asset_norm = normalize_asset(asset)
        config = await self.get_margin_config(account_id, asset_norm)
        if isinstance(config, dict) and "data" in config and isinstance(config["data"], dict):
            config = config["data"]
        config_id = (
            config.get("id")
            or config.get("configId")
            or config.get("_id")
        )
        if not config_id:
            raise ProprAPIError(404, f"Could not find margin configId for {asset_norm}")
        # Propr's margin-config PUT requires the full object, not just ``leverage``
        # (PR #1 review finding #2). Default marginMode to the existing value if
        # present, else fall back to ``cross``.
        margin_mode = "cross"
        if isinstance(config, dict):
            mm = config.get("marginMode") or config.get("margin_mode")
            if mm:
                margin_mode = str(mm)
        body = {
            "exchange": "hyperliquid",
            "asset": asset_norm,
            "marginMode": margin_mode,
            "leverage": int(leverage),
        }
        return await self._request(
            "PUT", f"/accounts/{account_id}/margin-config/{config_id}", json=body
        )

    async def place_order(self, **order_fields: Any) -> Dict[str, Any]:
        """Submit a single order via POST /accounts/{id}/orders.

        The caller passes only business fields (``asset``, ``side``, ``type``,
        ``quantity``, ...). This method injects the constants required by the
        API — a fresh ULID ``intentId``, ``exchange=hyperliquid``,
        ``productType=perp``, ``base=asset``, ``quote=USDC`` — and normalizes
        the asset for HIP-3 routing.
        """
        account_id = order_fields.pop("account_id", None) or order_fields.pop(
            "accountId", None
        )
        if not account_id:
            raise ValueError("place_order requires account_id")

        asset = order_fields.pop("asset", None)
        if not asset:
            raise ValueError("place_order requires asset")
        asset_norm = normalize_asset(asset)

        order: Dict[str, Any] = {
            "accountId": account_id,
            "intentId": str(ULID()),
            "exchange": "hyperliquid",
            "productType": "perp",
            "asset": asset_norm,
            "base": asset_norm,
            "quote": "USDC",
        }
        for key, value in order_fields.items():
            if value is None:
                continue
            if isinstance(value, Decimal):
                order[key] = format(value, "f")
            else:
                order[key] = value

        order.setdefault("reduceOnly", False)
        order.setdefault("closePosition", False)

        order_type = str(order.get("type", "market")).lower()
        if "timeInForce" not in order:
            order["timeInForce"] = "IOC" if order_type == "market" else "GTC"

        envelope = {"orders": [order]}
        log.debug(
            "place_order account=%s type=%s side=%s asset=%s qty=%s",
            account_id,
            order_type,
            order.get("side"),
            asset_norm,
            order.get("quantity"),
        )
        return await self._request(
            "POST", f"/accounts/{account_id}/orders", json=envelope
        )

    async def cancel_order(self, account_id: str, order_id: str) -> Dict[str, Any]:
        """Cancel a single order.

        A ``400`` response is treated as "already filled/cancelled" ONLY when
        :func:`_looks_like_already_done` confirms the body carries a terminal
        sentinel (``code 13053`` or ``status/state ∈ {cancelled, filled,
        expired}``). Anything else — including plain ``message: "bad_request"``
        — re-raises as :class:`ProprAPIError` so callers see the real error.
        """
        # review finding 1: don't blanket-swallow 400s; only soft-succeed on
        # confirmed terminal-state sentinels so genuine client errors surface.
        try:
            result = await self._request(
                "POST",
                f"/accounts/{account_id}/orders/{order_id}/cancel",
            )
        except ProprAPIError as exc:
            if exc.status == 400 and _looks_like_already_done(exc.body):
                log.info("cancel_order %s returned 400 (already done): %s", order_id, exc.body)
                return {"status": "already_done", "orderId": order_id}
            raise

        if _looks_like_already_done(result):
            log.info("cancel_order %s already done: %s", order_id, result)
            return {"status": "already_done", "orderId": order_id}
        return result

    # ------------------------------------------------------------------
    # Convenience shortcuts
    # ------------------------------------------------------------------
    async def market_buy(
        self,
        account_id: str,
        asset: str,
        qty: Decimal | str,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """Market buy — long entry (or short reduce if ``reduce_only``)."""
        return await self.place_order(
            account_id=account_id,
            asset=asset,
            type="market",
            side="buy",
            positionSide="short" if reduce_only else "long",
            timeInForce="IOC",
            quantity=_fmt_decimal(qty),
            reduceOnly=reduce_only,
        )

    async def market_sell(
        self,
        account_id: str,
        asset: str,
        qty: Decimal | str,
        reduce_only: bool = True,
    ) -> Dict[str, Any]:
        """Market sell — defaults to reduce-only (closes/reduces a long)."""
        return await self.place_order(
            account_id=account_id,
            asset=asset,
            type="market",
            side="sell",
            positionSide="long" if reduce_only else "short",
            timeInForce="IOC",
            quantity=_fmt_decimal(qty),
            reduceOnly=reduce_only,
        )

    async def limit_buy(
        self,
        account_id: str,
        asset: str,
        qty: Decimal | str,
        price: Decimal | str,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """Limit buy — GTC."""
        return await self.place_order(
            account_id=account_id,
            asset=asset,
            type="limit",
            side="buy",
            positionSide="short" if reduce_only else "long",
            timeInForce="GTC",
            quantity=_fmt_decimal(qty),
            price=_fmt_decimal(price),
            reduceOnly=reduce_only,
        )

    async def limit_sell(
        self,
        account_id: str,
        asset: str,
        qty: Decimal | str,
        price: Decimal | str,
        reduce_only: bool = True,
    ) -> Dict[str, Any]:
        """Limit sell — GTC, defaults to reduce-only."""
        return await self.place_order(
            account_id=account_id,
            asset=asset,
            type="limit",
            side="sell",
            positionSide="long" if reduce_only else "short",
            timeInForce="GTC",
            quantity=_fmt_decimal(qty),
            price=_fmt_decimal(price),
            reduceOnly=reduce_only,
        )

    async def close_position(
        self, account_id: str, asset: str, side: str
    ) -> Dict[str, Any]:
        """Fully close a position with ``closePosition=true, reduceOnly=true``.

        ``side`` is the *current* position side (``long`` or ``short``); the
        closing order uses the opposite taker side.

        Note (PR #1 review finding #1): the Propr API rejects ``quantity="0"``
        on a close, so we look up the actual open position and send its real
        quantity. Raises :class:`ProprAPIError` if no matching open position
        exists on this ``(asset, side)`` pair.
        """
        position_side = side.lower()
        if position_side not in ("long", "short"):
            raise ValueError("side must be 'long' or 'short'")

        asset_norm_upper = normalize_asset(asset).split(":")[-1].upper()
        positions = await self.get_positions(account_id)
        target: Optional[Dict[str, Any]] = None
        for p in positions:
            p_asset = str(p.get("asset", "")).split(":")[-1].upper()
            p_side = str(p.get("positionSide") or p.get("side") or "").lower()
            p_qty = str(p.get("quantity", p.get("qty", "0")))
            if p_asset == asset_norm_upper and p_side == position_side and p_qty not in ("0", "0.0", "0.00", ""):
                target = p
                break
        if target is None:
            raise ProprAPIError(
                404, f"No open {position_side} position on {asset} to close"
            )

        quantity = str(target.get("quantity", target.get("qty", "0")))
        taker_side = "sell" if position_side == "long" else "buy"
        return await self.place_order(
            account_id=account_id,
            asset=asset,
            type="market",
            side=taker_side,
            positionSide=position_side,
            timeInForce="IOC",
            quantity=quantity,
            reduceOnly=True,
            closePosition=True,
        )

    async def stop_loss(
        self,
        account_id: str,
        asset: str,
        qty: Decimal | str,
        trigger_price: Decimal | str,
        position_side: str,
    ) -> Dict[str, Any]:
        """Attach a stop-loss to an open position.

        For a ``long`` position the stop is a ``sell`` trigger; for a
        ``short`` position it's a ``buy`` trigger. Always reduce-only.
        """
        position_side = position_side.lower()
        if position_side not in ("long", "short"):
            raise ValueError("position_side must be 'long' or 'short'")
        taker_side = "sell" if position_side == "long" else "buy"
        return await self.place_order(
            account_id=account_id,
            asset=asset,
            type="stop_market",
            side=taker_side,
            positionSide=position_side,
            timeInForce="GTC",
            quantity=_fmt_decimal(qty),
            triggerPrice=_fmt_decimal(trigger_price),
            reduceOnly=True,
        )

    async def take_profit(
        self,
        account_id: str,
        asset: str,
        qty: Decimal | str,
        trigger_price: Decimal | str,
        position_side: str,
    ) -> Dict[str, Any]:
        """Attach a take-profit trigger to an open position."""
        position_side = position_side.lower()
        if position_side not in ("long", "short"):
            raise ValueError("position_side must be 'long' or 'short'")
        taker_side = "sell" if position_side == "long" else "buy"
        return await self.place_order(
            account_id=account_id,
            asset=asset,
            type="take_profit_market",
            side=taker_side,
            positionSide=position_side,
            timeInForce="GTC",
            quantity=_fmt_decimal(qty),
            triggerPrice=_fmt_decimal(trigger_price),
            reduceOnly=True,
        )


# ----------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------
def _looks_like_already_done(body: Any) -> bool:
    """Detect "order already filled/cancelled" 400 sentinels in a body dict.

    Tightened per review finding 1: the old ``message == "bad_request"``
    catch-all matched any generic 400. We now require one of:
      * ``code == 13053`` (documented Propr "order not active" code), OR
      * ``status`` / ``state`` equal to ``cancelled``, ``filled``, or
        ``expired`` (terminal order states).
    Anything else bubbles up as a real :class:`ProprAPIError`.
    """
    # review finding 1: narrow the already-done sentinel so genuine 400s
    # (e.g. bad payloads) aren't silently swallowed as soft-success.
    if not isinstance(body, dict):
        return False
    if body.get("code") == 13053:
        return True
    terminal = {"cancelled", "filled", "expired"}
    for key in ("status", "state"):
        val = body.get(key)
        if isinstance(val, str) and val.lower() in terminal:
            return True
    return False


def _as_list(data: Any) -> List[Dict[str, Any]]:
    """Unwrap list / {data:[...]} / {items:[...]} shapes into a plain list."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "items", "results", "orders", "positions", "trades"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def _fmt_decimal(value: Decimal | str | int | float) -> str:
    """Format a numeric value as a plain decimal string (no scientific notation)."""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (int, float)):
        return format(Decimal(str(value)), "f")
    return str(value)
