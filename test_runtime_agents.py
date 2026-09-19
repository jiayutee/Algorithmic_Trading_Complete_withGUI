"""Import/instantiation smoke tests for core/runtime/*.

Regression coverage for the Agent Monitor tab crash: core/runtime/price_agent.py
and core/runtime/supervisor.py used `SomeType | None` on subscripted typing
generics (e.g. `List[str] | None`) without `from __future__ import annotations`.
That syntax needs Python 3.10+ to evaluate at runtime; this project runs on
Python 3.9 (see CLAUDE.md), so importing either module raised
`TypeError: unsupported operand type(s) for |: '_GenericAlias' and 'NoneType'`
the moment ui/main_window.py's Agent Monitor tab tried to start the supervisor.
"""

import sys

import pytest


def test_price_agent_imports_and_constructs():
    from core.runtime.price_agent import PriceAgent

    agent = PriceAgent()
    assert agent.symbols == ["AAPL", "SPY"]
    agent2 = PriceAgent(symbols=["BTCUSDT"])
    assert agent2.symbols == ["BTCUSDT"]


def test_supervisor_imports_and_constructs():
    from core.runtime.supervisor import Supervisor

    supervisor = Supervisor()
    snap = supervisor.snapshot()
    assert "__meta__" in snap
    for key in ("portfolio", "news", "price", "stats"):
        assert key in snap


@pytest.mark.skipif(sys.version_info >= (3, 10), reason="only meaningful below 3.10")
def test_runtime_modules_have_no_bare_pep604_unions():
    """Belt-and-suspenders: every core/runtime/*.py module either has
    `from __future__ import annotations` or doesn't use `X | None`-style
    unions on subscripted generics, so this can't silently regress."""
    import ast
    from pathlib import Path

    runtime_dir = Path(__file__).parent / "core" / "runtime"
    for path in runtime_dir.glob("*.py"):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        has_future_import = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
            for node in tree.body
        )
        if has_future_import:
            continue
        # Without the future import, a `Subscript | X` annotation (e.g.
        # List[str] | None) would raise TypeError at def/class-body time on
        # Python < 3.10 -- fail loudly here instead of at import time.
        has_binop_or_on_subscript = any(
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.BitOr)
            and (isinstance(node.left, ast.Subscript) or isinstance(node.right, ast.Subscript))
            for node in ast.walk(tree)
        )
        assert not has_binop_or_on_subscript, (
            f"{path} uses a bare `X[...] | Y` annotation without "
            "`from __future__ import annotations` -- breaks on Python < 3.10"
        )
