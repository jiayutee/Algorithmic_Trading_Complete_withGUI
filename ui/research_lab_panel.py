"""
ui/research_lab_panel.py — Research Lab bottom-tab panel for AlgoTrader.

Provides three sub-tabs wired directly to the analytics in
``core.research_lab`` and ``core.volatility_lab``:

  **Strategy Lab**
    Sidebar strategy book (name + Sharpe) built from
    ``build_strategy_book()``.  Main area shows a chart selector that
    switches between drawdown, rolling Sharpe, trade P&L distribution,
    monthly-returns heatmap, and a year-by-year performance table.  Unit
    economics (win rate, expectancy, profit factor) appear inline in the
    control row.

  **Volatility Lab**
    Real-vs-shuffled rolling annualised vol chart plus regime tape
    (calm / normal / turbulent) in a two-row Plotly subplot.  A stats
    panel on the right shows excess kurtosis, ACF-by-lag, Ljung-Box
    p-value, same-sign rate, permutation-test lift, and a suggested
    position size from ``suggest_position_size()``.

  **Signal & Gate**
    Pass/fail gate verdict from ``evaluate_gate()``, with a per-check
    breakdown table and a one-paragraph human-readable summary.  A
    "Run Analysis" button (re)runs everything against the currently
    loaded data / backtest report.

All charts use QWebEngineView rendered via the same file-based Plotly
pattern as the main chart view (``template='plotly_dark'``, HTML written
to a temp file, loaded via ``QUrl.fromLocalFile``).  This avoids
QWebEngineView's 2 MB inline ``setHtml()`` cap.

Heavy analytics (strategy book, permutation test) run in a QThread
worker so the Qt event loop is never blocked.

Integration points in MainWindow
---------------------------------
Instantiate and add::

    self._research_lab_panel = ResearchLabPanel(
        data_loader=self.data_loader,
        strategy_manager=self.strategy_manager,
        parent_window=self,
    )
    self.bottom_tabs.addTab(self._research_lab_panel, "Research Lab")

Forward data and backtest results::

    # in _on_data_loaded
    if hasattr(self, '_research_lab_panel'):
        self._research_lab_panel.update_data(self.df)

    # in run_backtest, after successful results
    if hasattr(self, '_research_lab_panel') and results is not False:
        self._research_lab_panel.update_report(results)
"""

import math
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QColor

try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    _WEBENGINE_AVAILABLE = True
except ImportError:
    QWebEngineView = None
    _WEBENGINE_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    make_subplots = None
    _PLOTLY_AVAILABLE = False

from core.logger import logger


# ---------------------------------------------------------------------------
# Dark-theme palette — must mirror main_window.py
# ---------------------------------------------------------------------------
_BG_DARK  = "#0d1117"
_BG_MID   = "#161b22"
_BG_LIGHT = "#21262d"
_BORDER   = "#30363d"
_TEXT     = "#e6edf3"
_MUTED    = "#8b949e"
_BLUE     = "#58a6ff"
_GREEN    = "#3fb950"
_RED      = "#f85149"
_ORANGE   = "#f0883e"

# Compact chart dimensions that fit comfortably inside the ~270 px bottom pane.
_CHART_H      = 195
_CHART_MARGIN = dict(l=36, r=8, t=22, b=22)

# Regime-tape colour map
_REGIME_COLORS = {"calm": _BLUE, "normal": _GREEN, "turbulent": _RED}


# ---------------------------------------------------------------------------
# Internal helper — Plotly pane
# ---------------------------------------------------------------------------

class _PlotlyPane(QWidget):
    """QWebEngineView wrapper that renders Plotly figures via a temp file.

    Uses the same write-to-disk approach as ``update_plotly_view()`` in
    ``MainWindow`` to bypass the 2 MB ``setHtml()`` cap.  Falls back to a
    plain ``QLabel`` when QWebEngineWidgets is not installed.

    Args:
        placeholder: Message shown before the first chart is rendered.
        parent:      Qt parent widget.
    """

    def __init__(self, placeholder: str = "No data yet.", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if _WEBENGINE_AVAILABLE:
            self._view = QWebEngineView()
            self._show_text_page(placeholder)
        else:
            self._view = QLabel(placeholder)
            self._view.setAlignment(Qt.AlignCenter)
            self._view.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")

        layout.addWidget(self._view)
        self._tmp_file: Optional[str] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def render(self, fig) -> None:
        """Write *fig* to a temp HTML file and load it in the view.

        Silently no-ops when dependencies are missing or *fig* is None.
        """
        if not _WEBENGINE_AVAILABLE or not _PLOTLY_AVAILABLE:
            return
        if fig is None:
            self._show_text_page("No chart available.")
            return
        try:
            html = fig.to_html(include_plotlyjs=True, full_html=True)
            if self._tmp_file is None:
                fd, self._tmp_file = tempfile.mkstemp(
                    suffix=".html", prefix="algotrader_rl_"
                )
                os.close(fd)
            with open(self._tmp_file, "w", encoding="utf-8") as fh:
                fh.write(html)
            self._view.load(QUrl.fromLocalFile(self._tmp_file))
        except Exception as exc:
            logger.warning("_PlotlyPane.render failed: %s", exc)

    def show_message(self, text: str) -> None:
        """Replace the chart with a centred status message."""
        if _WEBENGINE_AVAILABLE:
            self._show_text_page(text)
        else:
            self._view.setText(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _show_text_page(self, text: str) -> None:
        """Render a minimal dark-background HTML page with centred text."""
        html = (
            f"<html><body style='background:{_BG_DARK};color:{_MUTED};"
            f"display:flex;align-items:center;justify-content:center;"
            f"height:100vh;margin:0;font-family:sans-serif;font-size:11px;"
            f"text-align:center;padding:8px;box-sizing:border-box;'>"
            f"{text}</body></html>"
        )
        self._view.setHtml(html)


# ---------------------------------------------------------------------------
# Background analytics worker
# ---------------------------------------------------------------------------

class _AnalysisWorker(QThread):
    """Run all Research Lab analytics in a background thread.

    Accepts a snapshot of the current state (OHLCV frame, backtest report,
    strategy manager) and a list of task names.  Emits ``results_ready``
    with a dict containing one key per completed task.  Individual task
    failures are captured as ``<task>_error`` keys so one broken strategy
    cannot abort the entire run.

    Supported tasks
    ---------------
    ``"strategy_book"``  — :func:`build_strategy_book`
    ``"strategy_lab"``   — drawdown, rolling Sharpe, distribution, monthly,
                           yearly, unit-economics analytics
    ``"volatility"``     — :func:`compute_volatility_clustering_report`
    ``"gate"``           — :func:`evaluate_gate`

    Args:
        df:               Current OHLCV DataFrame (may be ``None``).
        report:           Last backtest result dict (may be empty).
        strategy_manager: StrategyManager instance for ``build_strategy_book``.
        tasks:            Subset of supported task names to run.
    """

    results_ready = pyqtSignal(dict)
    progress      = pyqtSignal(str)

    def __init__(
        self,
        df: Optional[pd.DataFrame],
        report: dict,
        strategy_manager=None,
        tasks: Optional[List[str]] = None,
    ):
        super().__init__()
        self._df               = df
        self._report           = report or {}
        self._strategy_manager = strategy_manager
        self._tasks            = tasks or ["strategy_book", "strategy_lab", "volatility", "gate"]

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: C901 (acceptable complexity for a single orchestrating method)
        results: Dict[str, Any] = {}

        # ---- Derive return/date series --------------------------------
        # Prefer backtest-aligned returns; fall back to price pct_change.
        returns_list: List[float] = self._report.get("returns", [])
        dates_list:   List[str]   = self._report.get("dates", [])

        if not returns_list and self._df is not None and not self._df.empty:
            try:
                ret_series = self._df["Close"].pct_change().dropna()
                returns_list = ret_series.tolist()
                dates_list   = [d.strftime("%Y-%m-%d") for d in ret_series.index]
                logger.debug(
                    "_AnalysisWorker: derived %d returns from price data", len(returns_list)
                )
            except Exception as exc:
                logger.warning(
                    "_AnalysisWorker: could not derive returns from df: %s", exc
                )

        profit_per_trade    = self._report.get("profit_per_trade", [])
        total_asset_value   = self._report.get("total_asset_value", [])

        results["returns"] = returns_list
        results["dates"]   = dates_list

        # ---- Strategy Book -------------------------------------------
        if "strategy_book" in self._tasks:
            self.progress.emit("Building strategy book…")
            try:
                from core.research_lab import build_strategy_book  # noqa: PLC0415
                if self._strategy_manager is not None and self._df is not None \
                        and not self._df.empty:
                    book = build_strategy_book(self._strategy_manager, self._df)
                else:
                    book = []
                results["strategy_book"] = book
            except Exception as exc:
                logger.warning("_AnalysisWorker: strategy_book failed: %s", exc)
                results["strategy_book"] = []
                results["strategy_book_error"] = str(exc)

        # ---- Strategy Lab charts ------------------------------------
        if "strategy_lab" in self._tasks:
            self.progress.emit("Computing strategy lab analytics…")
            try:
                from core.research_lab import (  # noqa: PLC0415
                    compute_drawdown_series,
                    compute_rolling_sharpe,
                    monthly_returns_table,
                    trade_pnl_distribution,
                    unit_economics_per_trade,
                    year_by_year_table,
                )
                results["strategy_lab"] = {
                    "drawdown":          compute_drawdown_series(total_asset_value),
                    "rolling_sharpe":    compute_rolling_sharpe(returns_list),
                    "distribution":      trade_pnl_distribution(profit_per_trade),
                    "monthly":           monthly_returns_table(returns_list, dates_list),
                    "yearly":            year_by_year_table(returns_list, dates_list),
                    "unit_economics":    unit_economics_per_trade(profit_per_trade),
                    "dates":             dates_list,
                    "total_asset_value": total_asset_value,
                }
            except Exception as exc:
                logger.warning("_AnalysisWorker: strategy_lab failed: %s", exc)
                results["strategy_lab"] = {}
                results["strategy_lab_error"] = str(exc)

        # ---- Volatility Lab -----------------------------------------
        if "volatility" in self._tasks:
            self.progress.emit("Computing volatility clustering analytics…")
            try:
                from core.volatility_lab import compute_volatility_clustering_report  # noqa: PLC0415
                vol = compute_volatility_clustering_report(
                    returns_list, dates=dates_list or None
                )
                results["volatility"] = vol
            except Exception as exc:
                logger.warning("_AnalysisWorker: volatility failed: %s", exc)
                results["volatility"] = {}
                results["volatility_error"] = str(exc)

        # ---- Gate ---------------------------------------------------
        if "gate" in self._tasks:
            self.progress.emit("Evaluating gate…")
            try:
                from core.research_lab import evaluate_gate  # noqa: PLC0415
                results["gate"] = evaluate_gate(self._report)
            except Exception as exc:
                logger.warning("_AnalysisWorker: gate failed: %s", exc)
                results["gate"] = {
                    "passed":       False,
                    "checks":       [],
                    "verdict_text": f"Gate evaluation failed: {exc}",
                }

        self.results_ready.emit(results)


# ---------------------------------------------------------------------------
# Chart factory helpers
# ---------------------------------------------------------------------------

def _make_drawdown_fig(dates: List[str], drawdown: List[float]):
    """Return a compact Plotly drawdown area chart."""
    if not dates or not drawdown or not _PLOTLY_AVAILABLE:
        return None
    # Align lengths (dates may be shorter if derived from price data)
    n = min(len(dates), len(drawdown))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates[:n],
        y=drawdown[:n],
        fill="tozeroy",
        fillcolor="rgba(248,81,73,0.25)",
        line=dict(color=_RED, width=1.5),
        name="Drawdown %",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=_CHART_H,
        margin=_CHART_MARGIN,
        yaxis_title="DD %",
        showlegend=False,
    )
    return fig


def _make_rolling_sharpe_fig(dates: List[str], rolling_sharpe: List[float]):
    """Return a compact Plotly rolling-Sharpe line chart."""
    if not dates or not rolling_sharpe or not _PLOTLY_AVAILABLE:
        return None
    n = min(len(dates), len(rolling_sharpe))
    valid_x, valid_y = [], []
    for i in range(n):
        v = rolling_sharpe[i]
        if not (isinstance(v, float) and math.isnan(v)):
            valid_x.append(dates[i])
            valid_y.append(v)

    if not valid_x:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid_x,
        y=valid_y,
        line=dict(color=_BLUE, width=1.5),
        name="Rolling Sharpe (63d)",
    ))
    fig.add_hline(y=0, line=dict(color=_MUTED, dash="dash", width=1))
    fig.update_layout(
        template="plotly_dark",
        height=_CHART_H,
        margin=_CHART_MARGIN,
        yaxis_title="Sharpe",
        showlegend=False,
    )
    return fig


def _make_pnl_distribution_fig(distribution: dict):
    """Return a compact Plotly P&L histogram."""
    if not distribution or not _PLOTLY_AVAILABLE:
        return None
    edges  = distribution.get("bin_edges", [])
    counts = distribution.get("counts", [])
    if not edges or not counts:
        return None

    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    colors  = [_GREEN if c > 0 else _RED for c in centers]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centers,
        y=counts,
        marker_color=colors,
        name="Trade P&L",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=_CHART_H,
        margin=_CHART_MARGIN,
        xaxis_title="P&L ($)",
        yaxis_title="Trades",
        showlegend=False,
    )
    return fig


def _make_monthly_heatmap_fig(monthly: dict):
    """Return a Plotly heatmap of monthly returns.

    *monthly* is the dict returned by ``monthly_returns_table()``:
    integer year keys map to ``{month_abbr: pct_return}`` inner dicts.
    """
    if not monthly or not _PLOTLY_AVAILABLE:
        return None
    _MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Year keys may be np.int32 (from pandas groupby) or Python int; accept both.
    years = sorted(k for k in monthly if isinstance(k, (int, np.integer)))
    if not years:
        return None

    z_data: List[List] = []
    text_data: List[List[str]] = []
    for yr in years:
        row = []
        txt_row = []
        for m in _MONTHS:
            val = monthly[yr].get(m)
            row.append(val)
            txt_row.append(f"{val:.1f}%" if val is not None else "")
        z_data.append(row)
        text_data.append(txt_row)

    fig = go.Figure(go.Heatmap(
        x=_MONTHS,
        y=[str(y) for y in years],
        z=z_data,
        colorscale="RdYlGn",
        zmid=0,
        text=text_data,
        texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(
        template="plotly_dark",
        height=_CHART_H,
        margin=_CHART_MARGIN,
        xaxis_title="Month",
        yaxis_title="Year",
    )
    return fig


def _make_vol_regime_fig(vol_report: dict, dates: List[str]):
    """Return a two-row Plotly figure: rolling vol (top) + regime tape (bottom).

    Args:
        vol_report: Dict from :func:`compute_volatility_clustering_report`.
        dates:      ISO date strings aligned to the return series.
    """
    if not vol_report or not _PLOTLY_AVAILABLE or not make_subplots:
        return None

    ann_vol    = vol_report.get("ann_vol_series", [])
    tape       = vol_report.get("regime_tape", {})
    labels     = tape.get("labels", [])
    sh_labels  = tape.get("shuffled_labels", [])
    vol_dates  = dates or list(range(len(ann_vol)))

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=["Rolling Ann. Vol (21d)", "Regime Tape (real vs. shuffled)"],
        vertical_spacing=0.06,
    )

    # ---- Row 1: real vol ---
    valid_x_real = [vol_dates[i] for i, v in enumerate(ann_vol) if v is not None and i < len(vol_dates)]
    valid_y_real = [v for v in ann_vol if v is not None]
    if valid_x_real:
        fig.add_trace(go.Scatter(
            x=valid_x_real, y=valid_y_real,
            line=dict(color=_BLUE, width=1.5),
            name="Real vol",
        ), row=1, col=1)

    # ---- Row 2: regime tape (real) ---
    for regime, color in _REGIME_COLORS.items():
        x_real = [
            vol_dates[i] for i, lbl in enumerate(labels)
            if lbl == regime and i < len(vol_dates)
        ]
        if x_real:
            fig.add_trace(go.Scatter(
                x=x_real,
                y=[1.0] * len(x_real),
                mode="markers",
                marker=dict(symbol="square", size=5, color=color),
                name=f"real:{regime}",
                showlegend=True,
            ), row=2, col=1)

    # ---- Row 2: regime tape (shuffled) ---
    for regime, color in _REGIME_COLORS.items():
        x_sh = [
            vol_dates[i] for i, lbl in enumerate(sh_labels)
            if lbl == regime and i < len(vol_dates)
        ]
        if x_sh:
            fig.add_trace(go.Scatter(
                x=x_sh,
                y=[0.0] * len(x_sh),
                mode="markers",
                marker=dict(symbol="square", size=5, color=color, opacity=0.45),
                name=f"shuffled:{regime}",
                showlegend=False,
            ), row=2, col=1)

    fig.update_yaxes(title_text="Ann. Vol", row=1, col=1)
    fig.update_yaxes(
        tickvals=[0, 1], ticktext=["Shuffled", "Real"],
        row=2, col=1,
    )
    fig.update_layout(
        template="plotly_dark",
        height=_CHART_H,
        margin=_CHART_MARGIN,
        legend=dict(font=dict(size=9), orientation="h", y=-0.05),
    )
    return fig


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class ResearchLabPanel(QWidget):
    """Research Lab panel — three analytics sub-tabs in the bottom pane.

    Designed to be instantiated once in ``MainWindow.__init__`` and
    registered with ``bottom_tabs.addTab(panel, "Research Lab")``.

    State updates arrive via two public methods:

    ``update_data(df)``    — called by MainWindow after each data load.
    ``update_report(report)`` — called by MainWindow after a successful
                               backtest; triggers an automatic analysis run.

    Args:
        data_loader:      Application DataLoader (held for future use;
                          data itself arrives via ``update_data``).
        strategy_manager: StrategyManager passed to ``build_strategy_book``.
        parent_window:    Optional MainWindow reference (unused directly
                          but kept for forward-compatibility).
        parent:           Qt parent widget.
    """

    def __init__(
        self,
        data_loader,
        strategy_manager,
        parent_window=None,
        parent=None,
    ):
        super().__init__(parent)
        self._data_loader      = data_loader
        self._strategy_manager = strategy_manager
        self._parent_window    = parent_window

        # Analytics state
        self._df:               Optional[pd.DataFrame] = None
        self._report:           dict                   = {}
        self._analysis_results: dict                   = {}
        self._worker:           Optional[_AnalysisWorker] = None

        self._setup_ui()
        self._update_run_btn_state()

    # ------------------------------------------------------------------
    # Public API — called by MainWindow
    # ------------------------------------------------------------------

    def update_data(self, df: pd.DataFrame) -> None:
        """Update the panel's OHLCV data snapshot.

        Does not trigger automatic reanalysis so the panel stays idle until
        either a backtest completes or the user presses "Run Analysis".

        Args:
            df: New OHLCV DataFrame from the data loader.
        """
        self._df = df
        self._update_run_btn_state()

    def update_report(self, report: dict) -> None:
        """Update the panel with a fresh backtest report and run analysis.

        Called by ``MainWindow.run_backtest`` after a successful run.
        Triggers a full analytics refresh automatically so the Research Lab
        always reflects the latest backtest.

        Args:
            report: Dict returned by the backtester's ``run_backtest()``.
        """
        self._report = report or {}
        self._run_analysis()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._inner_tabs = QTabWidget()
        self._inner_tabs.addTab(self._build_strategy_lab(),  "Strategy Lab")
        self._inner_tabs.addTab(self._build_volatility_lab(), "Volatility Lab")
        self._inner_tabs.addTab(self._build_signal_gate(),   "Signal & Gate")
        root.addWidget(self._inner_tabs)

    # ------------------------------------------------------------------
    # Sub-tab builders
    # ------------------------------------------------------------------

    def _build_strategy_lab(self) -> QWidget:
        """Build the Strategy Lab sub-tab widget."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)

        # ---- Left sidebar: strategy book --------------------------------
        sidebar = QWidget()
        sidebar.setMinimumWidth(130)
        sidebar.setMaximumWidth(190)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_layout.setSpacing(2)

        sb_lbl = QLabel("Strategy Book")
        sb_lbl.setStyleSheet(
            f"color: {_MUTED}; font-size: 10px; font-weight: 600;"
        )
        sb_layout.addWidget(sb_lbl)

        self._strategy_book_table = QTableWidget(0, 3)
        self._strategy_book_table.setHorizontalHeaderLabels(["Strategy", "Sharpe", "Win%"])
        self._strategy_book_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._strategy_book_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self._strategy_book_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self._strategy_book_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._strategy_book_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._strategy_book_table.setAlternatingRowColors(True)
        self._strategy_book_table.verticalHeader().setVisible(False)
        sb_layout.addWidget(self._strategy_book_table)

        splitter.addWidget(sidebar)

        # ---- Right: chart area ------------------------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        # Control row: chart selector + inline unit economics
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(6)

        chart_lbl = QLabel("Chart:")
        chart_lbl.setStyleSheet(f"color: {_MUTED}; font-size: 10px;")
        ctrl_row.addWidget(chart_lbl)

        self._sl_chart_combo = QComboBox()
        self._sl_chart_combo.addItems([
            "Drawdown",
            "Rolling Sharpe",
            "P&L Distribution",
            "Monthly Returns",
            "Year-by-Year Table",
        ])
        self._sl_chart_combo.setFixedWidth(155)
        self._sl_chart_combo.currentTextChanged.connect(
            self._refresh_strategy_lab_chart
        )
        ctrl_row.addWidget(self._sl_chart_combo)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: {_BORDER};")
        ctrl_row.addWidget(sep)

        self._sl_ue_label = QLabel("Run a backtest to see unit economics.")
        self._sl_ue_label.setStyleSheet(f"color: {_MUTED}; font-size: 10px;")
        ctrl_row.addWidget(self._sl_ue_label)
        ctrl_row.addStretch()

        right_layout.addLayout(ctrl_row)

        # Stacked widget: Plotly pane (index 0) or year table (index 1)
        self._sl_stack = QStackedWidget()

        self._sl_plotly_pane = _PlotlyPane(
            "Run a backtest then click 'Run Analysis' in the Signal &amp; Gate tab."
        )
        self._sl_stack.addWidget(self._sl_plotly_pane)  # index 0

        self._sl_year_table = QTableWidget(0, 5)
        self._sl_year_table.setHorizontalHeaderLabels(
            ["Year", "Return %", "Benchmark %", "Sharpe", "Max DD %"]
        )
        for col in range(5):
            self._sl_year_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )
        self._sl_year_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._sl_year_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._sl_year_table.setAlternatingRowColors(True)
        self._sl_year_table.verticalHeader().setVisible(False)
        self._sl_stack.addWidget(self._sl_year_table)  # index 1

        right_layout.addWidget(self._sl_stack)
        splitter.addWidget(right)
        splitter.setSizes([150, 500])

        layout.addWidget(splitter)
        return tab

    def _build_volatility_lab(self) -> QWidget:
        """Build the Volatility Lab sub-tab widget."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)

        # ---- Left: rolling vol + regime tape chart ----------------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._vl_vol_pane = _PlotlyPane(
            "Load data then click 'Run Analysis' in the Signal &amp; Gate tab."
        )
        left_layout.addWidget(self._vl_vol_pane)
        splitter.addWidget(left)

        # ---- Right: stats panel ----------------------------------------
        right = QWidget()
        right.setMinimumWidth(190)
        right.setMaximumWidth(290)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(2)

        stats_lbl = QLabel("Volatility Statistics")
        stats_lbl.setStyleSheet(
            f"color: {_MUTED}; font-size: 10px; font-weight: 600;"
        )
        right_layout.addWidget(stats_lbl)

        self._vl_stats_text = QTextEdit()
        self._vl_stats_text.setReadOnly(True)
        self._vl_stats_text.setStyleSheet(
            f"background: {_BG_MID}; border: 1px solid {_BORDER}; border-radius: 4px;"
            f"color: {_TEXT}; font-family: 'SF Mono','Consolas',monospace; font-size: 10px;"
        )
        right_layout.addWidget(self._vl_stats_text)

        splitter.addWidget(right)
        splitter.setSizes([500, 210])

        layout.addWidget(splitter)
        return tab

    def _build_signal_gate(self) -> QWidget:
        """Build the Signal & Gate sub-tab widget."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar: status label + Run Analysis button
        toolbar = QHBoxLayout()
        self._gate_status_lbl = QLabel("Gate: no analysis run yet.")
        self._gate_status_lbl.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        toolbar.addWidget(self._gate_status_lbl)
        toolbar.addStretch()

        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.setFixedWidth(110)
        self._run_btn.setToolTip(
            "Recompute all Research Lab analytics against the current data and backtest report."
        )
        self._run_btn.clicked.connect(self._run_analysis)
        toolbar.addWidget(self._run_btn)
        layout.addLayout(toolbar)

        # Content: per-check table + verdict text
        content = QHBoxLayout()
        content.setSpacing(4)

        self._gate_check_table = QTableWidget(0, 3)
        self._gate_check_table.setHorizontalHeaderLabels(["Check", "Result", "Detail"])
        self._gate_check_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self._gate_check_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self._gate_check_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        self._gate_check_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._gate_check_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._gate_check_table.setAlternatingRowColors(True)
        self._gate_check_table.verticalHeader().setVisible(False)
        self._gate_check_table.setMaximumWidth(560)
        content.addWidget(self._gate_check_table)

        self._gate_verdict_text = QTextEdit()
        self._gate_verdict_text.setReadOnly(True)
        self._gate_verdict_text.setStyleSheet(
            f"background: {_BG_MID}; border: 1px solid {_BORDER}; border-radius: 4px;"
            f"color: {_TEXT}; font-size: 11px; padding: 4px;"
        )
        content.addWidget(self._gate_verdict_text)

        layout.addLayout(content)
        return tab

    # ------------------------------------------------------------------
    # Analysis runner
    # ------------------------------------------------------------------

    def _update_run_btn_state(self) -> None:
        """Enable the Run Analysis button only when there is data to analyse."""
        if not hasattr(self, "_run_btn"):
            return
        has_data = (self._df is not None and not self._df.empty) or bool(self._report)
        self._run_btn.setEnabled(has_data)

    def _run_analysis(self) -> None:
        """Launch the background analytics worker.

        Aborts silently if a worker is already running to avoid overlapping
        computations.  The "Run Analysis" button is disabled until the
        worker completes.
        """
        if self._worker and self._worker.isRunning():
            logger.debug("ResearchLabPanel: analysis already running — ignoring request.")
            return

        if not ((self._df is not None and not self._df.empty) or bool(self._report)):
            self._set_status("No data or backtest report available. Load data first.")
            return

        self._set_status("Running analysis…", color=_ORANGE)
        self._run_btn.setEnabled(False)

        self._worker = _AnalysisWorker(
            df=self._df,
            report=self._report,
            strategy_manager=self._strategy_manager,
        )
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.results_ready.connect(self._on_analysis_complete)
        self._worker.start()

    def _on_worker_progress(self, message: str) -> None:
        """Update status label while the worker progresses."""
        self._set_status(message, color=_ORANGE)

    def _on_analysis_complete(self, results: dict) -> None:
        """Populate all three sub-tabs from the completed analytics results."""
        self._analysis_results = results
        self._run_btn.setEnabled(True)

        self._populate_strategy_book(results.get("strategy_book", []))
        self._populate_strategy_lab(results.get("strategy_lab", {}))
        self._populate_volatility_lab(
            results.get("volatility", {}),
            results.get("dates", []),
        )
        self._populate_gate(results.get("gate", {}))

        err_keys = [k for k in results if k.endswith("_error")]
        if err_keys:
            self._set_status(
                f"Analysis complete (with {len(err_keys)} warning(s)).", color=_ORANGE
            )
        else:
            self._set_status("Analysis complete.", color=_GREEN)

    # ------------------------------------------------------------------
    # Sub-tab populators
    # ------------------------------------------------------------------

    def _populate_strategy_book(self, book: list) -> None:
        """Fill the strategy book sidebar table.

        Successful entries are shown with Sharpe and win-rate; failed entries
        appear in orange with an error indicator.
        """
        self._strategy_book_table.setRowCount(len(book))
        for row, entry in enumerate(book):
            name_item = QTableWidgetItem(entry.get("name", "?"))

            if "error" in entry:
                sharpe_item = QTableWidgetItem("ERR")
                wr_item     = QTableWidgetItem("—")
                for item in (name_item, sharpe_item, wr_item):
                    item.setForeground(QColor(_ORANGE))
                sharpe_item.setToolTip(entry["error"])
            else:
                sharpe = entry.get("sharpe", 0.0)
                wr     = entry.get("win_rate", 0.0)
                sharpe_item = QTableWidgetItem(f"{sharpe:.2f}")
                wr_item     = QTableWidgetItem(f"{wr:.1f}%")
                color = _GREEN if sharpe >= 0.3 else _RED
                sharpe_item.setForeground(QColor(color))

            sharpe_item.setTextAlignment(Qt.AlignCenter)
            wr_item.setTextAlignment(Qt.AlignCenter)

            self._strategy_book_table.setItem(row, 0, name_item)
            self._strategy_book_table.setItem(row, 1, sharpe_item)
            self._strategy_book_table.setItem(row, 2, wr_item)

    def _populate_strategy_lab(self, lab: dict) -> None:
        """Refresh all Strategy Lab charts and the year-by-year table."""
        if not lab:
            self._sl_plotly_pane.show_message("No strategy lab data — run a backtest first.")
            self._sl_ue_label.setText("—")
            return

        # Unit economics inline label
        ue = lab.get("unit_economics", {})
        wr  = ue.get("win_rate_pct", 0.0)
        exp = ue.get("expectancy")
        pf  = ue.get("profit_factor")
        ue_parts = [f"Win%: <b>{wr:.1f}%</b>"]
        if exp is not None:
            sign = "+" if exp >= 0 else ""
            ue_parts.append(f"Exp: <b>{sign}${exp:.2f}</b>")
        if pf is not None:
            ue_parts.append(f"PF: <b>{pf:.2f}</b>")
        self._sl_ue_label.setText("  |  ".join(ue_parts))
        self._sl_ue_label.setTextFormat(Qt.RichText)

        # Year-by-year table
        yearly = lab.get("yearly", [])
        self._sl_year_table.setRowCount(len(yearly))
        for row, yr_row in enumerate(yearly):
            ret_pct = yr_row.get("return_pct")
            bm_pct  = yr_row.get("benchmark_pct")
            sharpe  = yr_row.get("sharpe")
            dd_pct  = yr_row.get("max_drawdown_pct")

            year_item   = QTableWidgetItem(str(yr_row.get("year", "?")))
            ret_item    = QTableWidgetItem(
                f"{ret_pct:+.2f}%" if ret_pct is not None else "—"
            )
            bm_item     = QTableWidgetItem(
                f"{bm_pct:+.2f}%" if bm_pct is not None else "—"
            )
            sharpe_item = QTableWidgetItem(
                f"{sharpe:.2f}" if sharpe is not None else "—"
            )
            dd_item     = QTableWidgetItem(
                f"{dd_pct:.2f}%" if dd_pct is not None else "—"
            )

            if ret_pct is not None:
                ret_item.setForeground(QColor(_GREEN if ret_pct >= 0 else _RED))

            for item in (year_item, ret_item, bm_item, sharpe_item, dd_item):
                item.setTextAlignment(Qt.AlignCenter)

            self._sl_year_table.setItem(row, 0, year_item)
            self._sl_year_table.setItem(row, 1, ret_item)
            self._sl_year_table.setItem(row, 2, bm_item)
            self._sl_year_table.setItem(row, 3, sharpe_item)
            self._sl_year_table.setItem(row, 4, dd_item)

        # Render whichever chart is currently selected
        self._refresh_strategy_lab_chart(self._sl_chart_combo.currentText())

    def _refresh_strategy_lab_chart(self, chart_name: str) -> None:
        """Switch the Strategy Lab chart area to the selected chart type.

        Called both programmatically (after a new analysis completes) and
        by the combo-box ``currentTextChanged`` signal.
        """
        lab = self._analysis_results.get("strategy_lab", {})

        if chart_name == "Year-by-Year Table":
            self._sl_stack.setCurrentIndex(1)
            return

        # Switch to Plotly pane
        self._sl_stack.setCurrentIndex(0)

        if not lab:
            self._sl_plotly_pane.show_message(
                "No backtest data yet — run a backtest first."
            )
            return

        dates         = lab.get("dates", [])
        drawdown      = lab.get("drawdown", [])
        rolling_sharpe = lab.get("rolling_sharpe", [])
        distribution   = lab.get("distribution", {})
        monthly        = lab.get("monthly", {})

        if chart_name == "Drawdown":
            fig = _make_drawdown_fig(dates, drawdown)
        elif chart_name == "Rolling Sharpe":
            fig = _make_rolling_sharpe_fig(dates, rolling_sharpe)
        elif chart_name == "P&L Distribution":
            fig = _make_pnl_distribution_fig(distribution)
        elif chart_name == "Monthly Returns":
            fig = _make_monthly_heatmap_fig(monthly)
        else:
            fig = None

        if fig is not None:
            self._sl_plotly_pane.render(fig)
        else:
            self._sl_plotly_pane.show_message(
                "No data for this chart — run a backtest first."
            )

    def _populate_volatility_lab(self, vol: dict, dates: List[str]) -> None:
        """Refresh the Volatility Lab chart and stats panel."""
        if not vol:
            self._vl_vol_pane.show_message(
                "No volatility data — load price data and click 'Run Analysis'."
            )
            self._vl_stats_text.setPlainText(
                "No data yet.\n\nLoad price data and click 'Run Analysis'."
            )
            return

        # ---- Chart ---
        fig = _make_vol_regime_fig(vol, dates)
        if fig is not None:
            self._vl_vol_pane.render(fig)
        else:
            self._vl_vol_pane.show_message("Chart unavailable (Plotly not installed).")

        # ---- Stats text panel ---
        ek    = vol.get("excess_kurtosis", float("nan"))
        acf   = vol.get("acf_abs", {})
        lb    = vol.get("ljung_box", {})
        ssr   = vol.get("same_sign_rate", float("nan"))
        perm  = vol.get("permutation", {})

        def _fmt(val, fmt=".4f"):
            try:
                if isinstance(val, float) and math.isnan(val):
                    return "n/a"
                return format(val, fmt)
            except (TypeError, ValueError):
                return "n/a"

        lines = [
            f"Excess kurtosis:  {_fmt(ek, '.4f')}",
            f"  (0 = Gaussian; >0 = heavier tails)",
            "",
            "ACF of |returns|:",
        ]
        for lag in sorted(acf.keys()):
            lines.append(f"  lag {lag:>2d}:  {_fmt(acf[lag], '.4f')}")

        lb_stat = lb.get("statistic", float("nan"))
        lb_pval = lb.get("p_value", float("nan"))
        lines += [
            "",
            f"Ljung-Box (lag 22):",
            f"  statistic: {_fmt(lb_stat, '.4f')}",
            f"  p-value:   {_fmt(lb_pval, '.4f')}",
            f"  {'clustering confirmed (p<0.05)' if isinstance(lb_pval, float) and not math.isnan(lb_pval) and lb_pval < 0.05 else 'no significant clustering'}",
            "",
            f"Same-sign rate: {_fmt(ssr, '.4f')}",
            f"  (>0.5 momentum, <0.5 mean-reversion)",
            "",
            "Permutation test (ACF lag-1 |r|):",
            f"  observed:       {_fmt(perm.get('observed'), '.4f')}",
            f"  shuffled mean:  {_fmt(perm.get('shuffled_mean'), '.4f')}",
            f"  shuffled std:   {_fmt(perm.get('shuffled_std'), '.4f')}",
            f"  lift:           {_fmt(perm.get('lift_pts'), '+.2f')} pts",
            f"  p-value:        {_fmt(perm.get('p_value'), '.4f')}",
        ]

        # Position sizing (uses the same returns used by the vol report)
        returns_for_sizing = self._analysis_results.get("returns", [])
        if returns_for_sizing:
            try:
                from core.volatility_lab import suggest_position_size  # noqa: PLC0415
                capital = 100_000.0
                if self._report:
                    final_val = self._report.get("summary", {}).get("Final Value")
                    if final_val:
                        capital = float(final_val)
                sizing = suggest_position_size(returns_for_sizing, capital=capital)
                sf  = sizing.get("suggested_fraction", 0.0)
                sn  = sizing.get("suggested_notional", 0.0)
                v99 = sizing.get("var_99", float("nan"))
                cv  = sizing.get("cvar_99", float("nan"))
                lines += [
                    "",
                    f"Position Sizing (2% risk budget, 99% VaR):",
                    f"  VaR 99%:   {_fmt(v99, '.4f')}",
                    f"  CVaR 99%:  {_fmt(cv, '.4f')}",
                    f"  Fraction:  {sf * 100:.1f}%",
                    f"  Notional:  ${sn:,.0f}",
                ]
            except Exception as exc:
                lines.append(f"\nPosition sizing unavailable: {exc}")

        self._vl_stats_text.setPlainText("\n".join(lines))

    def _populate_gate(self, gate: dict) -> None:
        """Fill the Signal & Gate sub-tab from an evaluate_gate() result."""
        if not gate:
            self._gate_status_lbl.setText("Gate: no result — run a backtest first.")
            self._gate_check_table.setRowCount(0)
            self._gate_verdict_text.setPlainText(
                "Run a backtest and then click 'Run Analysis' to see the gate verdict."
            )
            return

        passed  = gate.get("passed", False)
        checks  = gate.get("checks", [])
        verdict = gate.get("verdict_text", "")

        verdict_color = _GREEN if passed else _RED
        verdict_word  = "PASS" if passed else "FAIL"
        self._gate_status_lbl.setText(f"Gate verdict: {verdict_word}")
        self._gate_status_lbl.setStyleSheet(
            f"color: {verdict_color}; font-size: 11px; font-weight: 600;"
        )

        self._gate_check_table.setRowCount(len(checks))
        for row, check in enumerate(checks):
            name_item   = QTableWidgetItem(check.get("name", "?"))
            ok          = check.get("passed", False)
            result_item = QTableWidgetItem("PASS" if ok else "FAIL")
            result_item.setForeground(QColor(_GREEN if ok else _RED))
            result_item.setTextAlignment(Qt.AlignCenter)
            detail_item = QTableWidgetItem(check.get("detail", ""))

            self._gate_check_table.setItem(row, 0, name_item)
            self._gate_check_table.setItem(row, 1, result_item)
            self._gate_check_table.setItem(row, 2, detail_item)

        self._gate_verdict_text.setPlainText(verdict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_status(self, message: str, color: str = _MUTED) -> None:
        """Update the gate status label; used for progress messages too."""
        if hasattr(self, "_gate_status_lbl"):
            self._gate_status_lbl.setText(message)
            self._gate_status_lbl.setStyleSheet(
                f"color: {color}; font-size: 11px;"
            )
