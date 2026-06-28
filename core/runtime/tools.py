"""MCP/stdio tool adapter and defensive fallbacks.

This module implements a minimal JSON-RPC over stdio client to start and
talk to MCP servers configured in .vscode/mcp.json. It also exposes a small
set of helper functions that fall back to local repo components when no MCP
server is available.

Design notes:
- Use LSP-style Content-Length framing for stdio JSON-RPC.
- Keep implementation defensive: timeouts, multiple method fallbacks.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

from core.news_sources import BraveSearchSource
from core.runtime.base import RuntimeContext
from core.logger import get_logger

logger = get_logger(__name__)

# Small, local helper to read .vscode/mcp.json if present
MCP_CONFIG_PATH = os.path.join(os.getcwd(), ".vscode", "mcp.json")


class MCPClient:
    """Minimal JSON-RPC over stdio client (Content-Length framed).

    Methods are intentionally permissive: list_tools() and call_tool()
    will try several likely RPC method names to interoperate with different
    MCP servers. If the subprocess cannot be started or the server doesn't
    support the methods, callers should expect exceptions or None results.
    """

    def __init__(self, cmd: List[str], env: Optional[Dict[str, str]] = None, timeout: float = 5.0):
        self.cmd = cmd
        self.env = os.environ.copy()
        if env:
            self.env.update(env)
        self.proc: Optional[subprocess.Popen] = None
        self._id = 1
        self._resp_cond = threading.Condition()
        self._responses: Dict[int, Any] = {}
        self._reader_thread: Optional[threading.Thread] = None
        self.timeout = timeout

    def start(self) -> bool:
        if self.proc:
            return True
        try:
            self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env)
        except Exception:
            self.proc = None
            return False

        # start reader thread
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()
        # perform the standard MCP initialization handshake so the server will
        # advertise tools before we try to call them.
        try:
            self.request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {"name": "algorithmic-trading-runtime", "version": "1.0.0"},
                    "capabilities": {},
                },
                timeout=self.timeout,
            )
            self.notify("notifications/initialized", {})
        except Exception:
            # Some servers may not require an explicit initialized notification.
            # We still keep the process alive so fallback paths can continue.
            pass
        return True

    def stop(self) -> None:
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        finally:
            self.proc = None

    def _reader(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        buf = b""
        while self.proc and self.proc.stdout and not self.proc.stdout.closed:
            try:
                # Read headers
                header = b""
                # read until blank line
                while True:
                    line = self.proc.stdout.readline()
                    if not line:
                        return
                    header += line
                    if header.endswith(b"\r\n\r\n") or header.endswith(b"\n\n"):
                        break
                # parse Content-Length
                header_text = header.decode(errors="ignore")
                length = 0
                for hline in header_text.splitlines():
                    if hline.lower().startswith("content-length:"):
                        try:
                            length = int(hline.split(":", 1)[1].strip())
                        except Exception:
                            length = 0
                if length <= 0:
                    # nothing to read
                    continue
                payload = self.proc.stdout.read(length)
                if not payload:
                    return
                try:
                    msg = json.loads(payload.decode())
                except Exception:
                    continue
                # store response if it has id
                if isinstance(msg, dict) and "id" in msg:
                    with self._resp_cond:
                        self._responses[msg.get("id")] = msg
                        self._resp_cond.notify_all()
            except Exception:
                return

    def _send(self, payload: Dict[str, Any]) -> None:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("MCP process not started")
        raw = json.dumps(payload, ensure_ascii=False).encode()
        header = f"Content-Length: {len(raw)}\r\n\r\n".encode()
        try:
            self.proc.stdin.write(header + raw)
            self.proc.stdin.flush()
        except Exception as e:
            raise RuntimeError(f"failed to write to mcp stdin: {e}")

    def request(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Any:
        if not self.proc:
            raise RuntimeError("MCP process not started")
        mid = self._id
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}
        self._send(payload)
        timeout = timeout or self.timeout
        end = time.time() + timeout
        with self._resp_cond:
            while time.time() < end:
                if mid in self._responses:
                    return self._responses.pop(mid)
                remaining = end - time.time()
                if remaining <= 0:
                    break
                self._resp_cond.wait(timeout=remaining)
        raise TimeoutError(f"timeout waiting for response to {method}")

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._send(payload)


# High-level module helpers
_global_mcp: Optional[MCPClient] = None


def _load_mcp_config() -> Dict[str, Any]:
    if not os.path.exists(MCP_CONFIG_PATH):
        return {}
    try:
        with open(MCP_CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def start_mcp_server(server_key: str = "duckduckgo-search") -> Optional[MCPClient]:
    """Start a configured MCP server by key and return an MCPClient or None."""
    global _global_mcp
    cfg = _load_mcp_config().get("servers", {})
    entry = cfg.get(server_key)
    if not entry:
        return None
    cmd = []
    cmd.append(entry.get("command"))
    cmd.extend(entry.get("args", []))
    env = entry.get("env") or {}
    try:
        client = MCPClient(cmd, env=env)
        ok = client.start()
        if ok:
            _global_mcp = client
            # try an initialize if server supports it
            try:
                client.notify("initialize", {})
            except Exception:
                pass
            return client
    except Exception:
        return None
    return None


def get_mcp_client() -> Optional[MCPClient]:
    return _global_mcp


def list_tools(timeout: float = 2.0) -> Dict[str, Any]:
    """Attempt to list tools from an MCP server. Returns {'ok':bool, 'tools':...}.

    Tries multiple likely RPC names and falls back to an empty result.
    """
    client = get_mcp_client()
    if not client:
        return {"ok": False, "error": "mcp_not_started"}
    methods = ["tools/list", "tool/list", "mcp.listTools", "tool/listTools", "listTools"]
    for m in methods:
        try:
            resp = client.request(m, {}, timeout=timeout)
            # response may be full JSON-RPC envelope
            if isinstance(resp, dict) and "result" in resp:
                return {"ok": True, "tools": resp["result"]}
            return {"ok": True, "tools": resp}
        except Exception:
            continue
    return {"ok": False, "error": "no_list_method"}


def call_tool(name: str, input: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Dict[str, Any]:
    client = get_mcp_client()
    if not client:
        return {"ok": False, "error": "mcp_not_started"}
    # try a few likely method names for invoking tools
    methods = ["tools/call", "tool/call", "tool/invoke", "tools.call", "invokeTool", "tool.execute"]
    params = {"name": name, "input": input or {}}
    for m in methods:
        try:
            resp = client.request(m, params, timeout=timeout)
            if isinstance(resp, dict) and "result" in resp:
                return {"ok": True, "result": resp["result"]}
            return {"ok": True, "result": resp}
        except Exception:
            continue
    return {"ok": False, "error": "no_invoke_method"}


# Fallback local helpers (previous behavior)
def get_portfolio() -> Dict[str, Any]:
    try:
        from core.broker_manager import BrokerManager

        bm = BrokerManager()
        return {"ok": True, "portfolio": bm.get_portfolio()}
    except Exception as e:
        logger.warning("get_portfolio failed: %s", e)
        return {"ok": False, "error": "broker_manager unavailable"}


def get_latest_price(symbol: str) -> Dict[str, Any]:
    # try MCP search tool for prices if available
    m = get_mcp_client()
    if m:
        try:
            # Some servers expose a price tool; attempt a generic call
            resp = call_tool("price.get", {"symbol": symbol}, timeout=3.0)
            if resp.get("ok"):
                return {"ok": True, "symbol": symbol, "price": resp.get("result")}
        except Exception as e:
            logger.debug("MCP price.get failed for %s: %s", symbol, e)
    try:
        from core.data_loader import DataLoader

        dl = DataLoader()
        price = dl.get_latest_price(symbol)
        return {"ok": True, "symbol": symbol, "price": price}
    except Exception as e:
        logger.warning("get_latest_price(%s) failed: %s", symbol, e)
        return {"ok": False, "error": "data_loader unavailable"}


def fetch_recent_news(limit: int = 10) -> Dict[str, Any]:
    query = os.getenv("NEWS_SEARCH_QUERY", "latest stock market news").strip() or "latest stock market news"

    brave_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip() or os.getenv("BRAVE_API_KEY", "").strip()
    if brave_key:
        try:
            brave_source = BraveSearchSource(api_key=brave_key)
            brave_items = brave_source.fetch(query, limit=limit)
            if brave_items:
                return {"ok": True, "news": brave_items, "source": "brave", "query": query}
        except Exception as e:
            logger.debug("Brave news fetch failed: %s", e)

    # Prefer MCP-powered web search if available
    m = get_mcp_client()
    if m:
        try:
            resp = call_tool("search", {"query": query, "max_results": limit}, timeout=5.0)
            if resp.get("ok"):
                return {"ok": True, "news": resp.get("result"), "source": "mcp_ddg", "query": query}
        except Exception as e:
            logger.debug("MCP search failed: %s", e)
    try:
        from core.news_pipeline import NewsPipeline

        pipeline = NewsPipeline.from_env()
        items = pipeline.fetch_news_items(query, limit=limit)
        return {"ok": True, "news": items, "source": "pipeline", "query": query}
    except Exception as e:
        logger.warning("fetch_recent_news failed: %s", e)
        return {"ok": False, "error": "news_pipeline unavailable"}


def store_news(items: List[Dict[str, Any]], writer: Optional[Any] = None) -> Dict[str, Any]:
    if writer is not None:
        try:
            result = writer.submit_news_items(items)
            if isinstance(result, dict):
                return result
            return {"ok": True, "result": result}
        except Exception as exc:
            return {"ok": False, "error": f"sql_writer unavailable: {exc}"}
    try:
        from core.news_store import NewsStore

        ns = NewsStore()
        if hasattr(ns, "add_items"):
            inserted = ns.add_items(items)
            return {"ok": True, "inserted": inserted}
        if hasattr(ns, "insert_many"):
            inserted = ns.insert_many(items)
            return {"ok": True, "inserted": inserted}
        return {"ok": False, "error": "news_store_missing_insert_method"}
    except Exception as e:
        logger.warning("store_news failed: %s", e)
        return {"ok": False, "error": "news_store unavailable"}


def place_order(ctx: Optional["RuntimeContext"], order: Dict[str, Any]) -> Dict[str, Any]:
    """Place an order if policy permits. Expects a RuntimeContext to be provided.

    This function is defensive: if ctx or ctx.policy is missing, it refuses autonomous
    placement.
    """
    try:
        if ctx is None or getattr(ctx, "policy", None) is None:
            return {"ok": False, "error": "no_policy"}
        policy = ctx.policy
        if policy.require_trade_approval or not policy.autonomy_enabled.get("portfolio", False):
            return {"ok": False, "error": "trade_requires_approval"}
        # optional value checks
        value = float(order.get("value", 0))
        if value > policy.max_order_value:
            return {"ok": False, "error": "order_value_exceeds_policy"}
        from core.broker_manager import BrokerManager

        bm = BrokerManager()
        res = bm.place_order(order)
        return {"ok": True, "result": res}
    except Exception as e:
        return {"ok": False, "error": f"place_order_failed: {e}"}

