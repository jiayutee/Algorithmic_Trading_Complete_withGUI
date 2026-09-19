"""Tiny local experiment log (Phase 11.3): one SQLite file, no server, no new infrastructure.

Every training / evaluation run records what was run and how it did, so "which settings gave
the 0.73 volatility AUC?" is a query instead of an archaeology project:

    from core.experiment_log import ExperimentLog
    log = ExperimentLog()                                   # training_ground/results/experiments.sqlite3
    log.log_run(name="H5 volatility", model_type="lightgbm", params={...}, metrics={"auc": 0.73})
    log.list_runs(model_type="lightgbm")                    # newest first
    log.best("auc", n=3)                                    # top runs by a metric
    log.compare([4, 7])                                     # side-by-side DataFrame

Command line:  python -m core.experiment_log list | show ID | best METRIC | compare ID ID ...

Each run stores: timestamp (UTC), git commit (+ whether the tree had uncommitted changes),
model type, hyperparameters, metrics, dataset description, tags and free-text notes.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(_REPO_ROOT, "training_ground", "results", "experiments.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    name        TEXT NOT NULL,
    model_type  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'completed',
    git_commit  TEXT,
    git_dirty   INTEGER,
    params      TEXT NOT NULL DEFAULT '{}',
    metrics     TEXT NOT NULL DEFAULT '{}',
    dataset     TEXT NOT NULL DEFAULT '{}',
    tags        TEXT NOT NULL DEFAULT '[]',
    notes       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_type);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
"""


def default_path() -> str:
    """Log location: EXPERIMENT_LOG_PATH if set (tests, other machines), else the repo default."""
    return os.environ.get("EXPERIMENT_LOG_PATH") or DEFAULT_PATH


def git_state(cwd: str = _REPO_ROOT) -> Dict[str, Any]:
    """Current commit hash and whether there are uncommitted changes. Never raises."""
    def run(*args):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5).stdout.strip()
    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain"))
        return {"commit": commit or None, "dirty": dirty if commit else None}
    except Exception:  # noqa: BLE001 -- not a git checkout, git missing, timeout...
        return {"commit": None, "dirty": None}


def _jsonable(obj: Any) -> Any:
    """Make numpy/pandas scalars, NaN and paths JSON-safe."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    try:
        f = float(obj)
    except (TypeError, ValueError):
        return str(obj)
    return None if f != f or f in (float("inf"), float("-inf")) else f


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


class ExperimentLog:
    def __init__(self, path: Optional[str] = None):
        self.path = path or default_path()
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ writing

    def log_run(self, *, name: str, model_type: str, params: Optional[dict] = None,
                metrics: Optional[dict] = None, dataset: Optional[dict] = None,
                tags: Optional[Iterable[str]] = None, notes: str = "", status: str = "completed",
                git: Optional[dict] = None) -> int:
        """Record one run and return its id."""
        g = git if git is not None else git_state()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO runs (created_at, name, model_type, status, git_commit, git_dirty, params, metrics, dataset, tags, notes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), name, model_type, status,
                 g.get("commit"), None if g.get("dirty") is None else int(bool(g["dirty"])),
                 json.dumps(_jsonable(params or {})), json.dumps(_jsonable(metrics or {})),
                 json.dumps(_jsonable(dataset or {})), json.dumps(sorted(set(tags or []))), notes or ""),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    # ------------------------------------------------------------------ reading

    @staticmethod
    def _row(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for k in ("params", "metrics", "dataset", "tags"):
            d[k] = json.loads(d[k] or ("[]" if k == "tags" else "{}"))
        d["git_dirty"] = None if d["git_dirty"] is None else bool(d["git_dirty"])
        return d

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return self._row(row) if row else None

    def list_runs(self, *, model_type: Optional[str] = None, name_contains: Optional[str] = None,
                  tag: Optional[str] = None, since: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        sql, args = "SELECT * FROM runs WHERE 1=1", []
        if model_type:
            sql += " AND model_type = ?"; args.append(model_type)
        if name_contains:
            sql += " AND name LIKE ?"; args.append(f"%{name_contains}%")
        if tag:
            sql += " AND tags LIKE ?"; args.append(f'%"{tag}"%')
        if since:
            sql += " AND created_at >= ?"; args.append(since)
        sql += " ORDER BY id DESC LIMIT ?"; args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._row(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])

    def best(self, metric: str, *, n: int = 5, higher_is_better: bool = True,
             model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Top runs by a metric (nested metrics use dots, e.g. ``walk_forward.auc``). Runs lacking it are skipped."""
        scored = []
        for run in self.list_runs(model_type=model_type, limit=10_000):
            v = _flatten(run["metrics"]).get(metric)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                scored.append((v, run))
        scored.sort(key=lambda t: t[0], reverse=higher_is_better)
        return [r for _, r in scored[:n]]

    def to_frame(self, **filters) -> pd.DataFrame:
        """One row per run; params and metrics flattened into ``param.*`` / ``metric.*`` columns."""
        rows = []
        for r in self.list_runs(**{"limit": 10_000, **filters}):
            row = {k: r[k] for k in ("id", "created_at", "name", "model_type", "status", "git_commit", "git_dirty", "notes")}
            row.update({f"param.{k}": v for k, v in _flatten(r["params"]).items()})
            row.update({f"metric.{k}": v for k, v in _flatten(r["metrics"]).items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def compare(self, run_ids: Iterable[int]) -> pd.DataFrame:
        """Side-by-side view of chosen runs: rows = parameters/metrics, columns = run ids."""
        cols = {}
        for rid in run_ids:
            r = self.get_run(rid)
            if r is None:
                continue
            cols[rid] = {**{f"param.{k}": v for k, v in _flatten(r["params"]).items()},
                         **{f"metric.{k}": v for k, v in _flatten(r["metrics"]).items()}}
        return pd.DataFrame(cols).sort_index()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ----------------------------------------------------------------------------- CLI

def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m core.experiment_log", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list"); ls.add_argument("--model"); ls.add_argument("--name"); ls.add_argument("--limit", type=int, default=20)
    sh = sub.add_parser("show"); sh.add_argument("id", type=int)
    bs = sub.add_parser("best"); bs.add_argument("metric"); bs.add_argument("--n", type=int, default=5); bs.add_argument("--lowest", action="store_true")
    cp = sub.add_parser("compare"); cp.add_argument("ids", type=int, nargs="+")
    args = ap.parse_args(argv)
    log = ExperimentLog(args.path)
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    if args.cmd == "list":
        runs = log.list_runs(model_type=args.model, name_contains=args.name, limit=args.limit)
        for r in runs:
            key = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(_flatten(r["metrics"]).items())[:3])
            print(f"#{r['id']:<4} {r['created_at']}  {r['model_type']:<14} {r['name'][:38]:<38} {r['git_commit'][:7] if r['git_commit'] else '-------'}"
                  f"{'*' if r['git_dirty'] else ' '} {key}")
        print(f"({len(runs)} shown of {log.count()})")
    elif args.cmd == "show":
        r = log.get_run(args.id)
        print(json.dumps(r, indent=2) if r else f"no run #{args.id}")
        return 0 if r else 1
    elif args.cmd == "best":
        for r in log.best(args.metric, n=args.n, higher_is_better=not args.lowest):
            print(f"#{r['id']:<4} {_flatten(r['metrics'])[args.metric]:>10.4f}  {r['name']}")
    elif args.cmd == "compare":
        print(log.compare(args.ids).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
