"""
core/design_tokens.py

One source of truth for the look of BOTH front ends (design language v2):

* ``COLOR``      -- every color the UI may use, by role (surface, text, accent, up/down, warn, account modes).
* ``FONT_UI`` / ``FONT_MONO`` -- font stacks: sans for UI text, mono with tabular figures for numbers.
* ``SIZE``, ``SPACE``, ``RADIUS`` -- type scale (px), spacing steps (px), corner radii (px).

Consumers:
* desktop: ``qt_stylesheet()`` is the global Qt stylesheet of ``ui/main_window.py``;
* web: ``css_root()`` is injected into the Dash page as CSS variables (``dash_app/app.py``) and used by ``assets/theme.css``;
* charts: ``core/chart_builder.THEME`` reads its colors from here.

``tests/test_design_tokens.py`` checks that the generated stylesheets only use colors defined here.
"""

from __future__ import annotations

from typing import Dict, List

COLOR: Dict[str, str] = {
    # surfaces, darkest first
    "surface_base": "#0d1117",       # window / page background
    "surface_panel": "#161b22",      # cards, bars, inputs
    "surface_raised": "#21262d",     # secondary buttons, neutral chips
    "surface_hover": "#30363d",
    "surface_alt_row": "#1a2130",    # zebra rows
    "surface_selected": "#1f3d6b",   # selected row / active workspace
    "border": "#30363d",
    "border_dim": "#21262d",
    # text
    "text_primary": "#e6edf3",
    "text_secondary": "#c9d1d9",
    "text_muted": "#8b949e",         # lowest contrast allowed for readable text
    "text_disabled": "#6e7681",
    "text_on_fill": "#ffffff",
    # accent
    "accent": "#58a6ff",
    "accent_soft": "#a5d6ff",
    "accent_strong": "#1f6feb",
    # direction: text shades for dark surfaces, fill shades for buttons with white text
    "up": "#3fb950",
    "up_fill": "#238636",
    "up_border": "#2ea043",
    "up_bg": "#0f2818",
    "up_bg_text": "#7ee787",
    "down": "#ff7b72",
    "down_strong": "#f85149",        # chart candles, legacy loss color
    "down_fill": "#da3633",
    "down_bg": "#2d1111",
    "down_bg_text": "#ffa198",
    # status
    "warn": "#e3b341",
    "warn_fill": "#d29922",
    "warn_bg": "#3d2e00",
    "pending": "#f0883e",
    # account modes (see core/trading_status.py)
    "mode_paper_bg": "#0f3d3a",
    "mode_paper_fg": "#56d4c9",
    "mode_live_bg": "#b62324",
}

FONT_UI: List[str] = ["IBM Plex Sans", "SF Pro Text", "Helvetica Neue", "Segoe UI", "Arial", "sans-serif"]
FONT_MONO: List[str] = ["IBM Plex Mono", "SF Mono", "Menlo", "Consolas", "monospace"]

SIZE: Dict[str, int] = {"caption": 12, "body": 13, "heading": 15, "price": 16, "title": 22}   # px; 12 is the floor
SPACE: List[int] = [4, 8, 12, 16, 24, 32]
RADIUS: Dict[str, int] = {"chip": 4, "control": 6, "pill": 14, "dialog": 10}


def css_font(families: List[str]) -> str:
    """A CSS font-family value: names with spaces quoted."""
    return ", ".join(f"'{f}'" if " " in f else f for f in families)


def css_root() -> str:
    """``:root { --color-surface-base: #0d1117; ... }`` for the web app."""
    lines = [f"  --color-{k.replace('_', '-')}: {v};" for k, v in COLOR.items()]
    lines += [f"  --size-{k}: {v}px;" for k, v in SIZE.items()]
    lines += [f"  --space-{i}: {v}px;" for i, v in enumerate(SPACE, start=1)]
    lines += [f"  --radius-{k}: {v}px;" for k, v in RADIUS.items()]
    lines += [f"  --font-ui: {css_font(FONT_UI)};", f"  --font-mono: {css_font(FONT_MONO)};"]
    return ":root {\n" + "\n".join(lines) + "\n}\n"


def qt_stylesheet() -> str:
    """The desktop app's global Qt stylesheet, built from the tokens."""
    c, s = COLOR, SIZE
    return f"""
        QMainWindow, QWidget {{ background: {c['surface_base']}; color: {c['text_primary']}; }}
        QGroupBox {{ border: 1px solid {c['border']}; border-radius: {RADIUS['control']}px; margin-top: 8px;
                    font-size: {s['caption']}px; font-weight: 600; color: {c['text_muted']}; padding: 6px 4px 4px 4px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
        QComboBox {{ background: {c['surface_panel']}; border: 1px solid {c['border']}; border-radius: {RADIUS['chip']}px;
                    padding: 3px 6px; color: {c['text_primary']}; font-size: {s['caption']}px; min-height: 22px; }}
        QCheckBox {{ color: {c['text_primary']}; font-size: {s['caption']}px; spacing: 6px; }}
        QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {c['text_muted']}; border-radius: 3px;
                               background: {c['surface_panel']}; }}
        QCheckBox::indicator:hover {{ border-color: {c['accent']}; }}
        QCheckBox::indicator:checked {{ background: {c['accent_strong']}; border-color: {c['accent']}; }}
        QComboBox::drop-down {{ border: none; width: 18px; }}
        QComboBox QAbstractItemView {{ background: {c['surface_panel']}; border: 1px solid {c['border']}; color: {c['text_primary']}; }}
        QLineEdit {{ background: {c['surface_panel']}; border: 1px solid {c['border']}; border-radius: {RADIUS['chip']}px;
                    padding: 3px 6px; color: {c['text_primary']}; font-size: {s['caption']}px; min-height: 22px; }}
        QLineEdit:focus {{ border-color: {c['accent']}; }}
        QPushButton {{ background: {c['surface_raised']}; border: 1px solid {c['border']}; border-radius: {RADIUS['chip']}px;
                      padding: 4px 10px; color: {c['text_primary']}; font-size: {s['caption']}px; min-height: 24px; }}
        QPushButton:hover {{ background: {c['surface_hover']}; border-color: {c['accent']}; }}
        QPushButton:pressed {{ background: {c['surface_panel']}; }}
        QPushButton:disabled {{ color: {c['text_disabled']}; border-color: {c['border_dim']}; }}
        QPushButton:checked {{ background: {c['surface_selected']}; border-color: {c['accent']}; color: {c['text_on_fill']}; }}
        QTabWidget::pane {{ border: 1px solid {c['border']}; border-radius: {RADIUS['chip']}px; }}
        QTabBar::tab {{ background: {c['surface_panel']}; border: 1px solid {c['border']}; padding: 5px 12px;
                       color: {c['text_muted']}; font-size: {s['caption']}px; border-bottom: none;
                       border-radius: {RADIUS['chip']}px {RADIUS['chip']}px 0 0; }}
        QTabBar::tab:selected {{ background: {c['surface_base']}; color: {c['text_primary']}; border-bottom: 2px solid {c['accent']}; }}
        QTableWidget {{ background: {c['surface_base']}; alternate-background-color: {c['surface_alt_row']};
                       gridline-color: {c['border_dim']}; color: {c['text_primary']}; border: none; font-size: {s['caption']}px; }}
        QTableWidget::item:selected {{ background: {c['surface_selected']}; color: {c['text_on_fill']}; }}
        QHeaderView::section {{ background: {c['surface_panel']}; color: {c['text_muted']}; border: none;
                               border-bottom: 1px solid {c['border']}; padding: 4px 6px; font-size: {s['caption']}px;
                               font-weight: 600; }}
        QTextEdit {{ background: {c['surface_panel']}; border: 1px solid {c['border']}; border-radius: {RADIUS['chip']}px;
                    color: {c['text_primary']}; font-family: 'Menlo'; font-size: {s['caption']}px; }}
        QScrollBar:vertical {{ background: {c['surface_base']}; width: 8px; }}
        QScrollBar::handle:vertical {{ background: {c['surface_hover']}; border-radius: {RADIUS['chip']}px; min-height: 20px; }}
        QSplitter::handle {{ background: {c['border']}; width: 1px; height: 1px; }}
        QStatusBar {{ background: {c['surface_panel']}; border-top: 1px solid {c['border']}; color: {c['text_muted']};
                     font-size: {s['caption']}px; }}
        QLabel {{ color: {c['text_primary']}; font-size: {s['caption']}px; }}
        QPushButton#liveBtn {{ background: {c['up_bg']}; border-color: {c['up_border']}; color: {c['up']}; font-weight: 600; }}
        QPushButton#liveBtn:hover {{ background: {c['up_fill']}; color: {c['text_on_fill']}; }}
        QPushButton#workspaceBtn {{ background: transparent; border: none; color: {c['text_secondary']};
                                   font-size: {s['body']}px; padding: 4px 12px; min-height: 28px; }}
        QPushButton#workspaceBtn:checked {{ background: {c['surface_selected']}; color: {c['text_on_fill']}; font-weight: 600; }}
    """


def palette() -> set:
    """Every color value defined here (lower-case hex)."""
    return {v.lower() for v in COLOR.values()}
