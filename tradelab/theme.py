"""Оформление терминала: тёмная сетка в духе профессиональных платформ (янтарный акцент, плотные панели, моноширинные числа).
Только CSS и небольшие HTML-блоки; логика данных не затрагивается."""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .ui import C

SNAP = Path(__file__).resolve().parent.parent / "assets" / "snapshot"

CSS = f"""
<style>
:root {{
  --bg:{C['bg']}; --panel:#10151D; --panel2:#151C27; --line:#222C3B; --line2:#2E3A4E; --amber:{C['accent']}; --amber-dim:#8A5A10;
  --text:#D7DEE8; --muted:#7D8898; --up:{C['up']}; --down:{C['down']}; --blue:{C['blue']};
  --mono:'JetBrains Mono','IBM Plex Mono','SF Mono',Consolas,'Liberation Mono',monospace;
}}
html, body, [data-testid="stAppViewContainer"], .stApp {{background: var(--bg); font-family: var(--mono);}}
[data-testid="stHeader"] {{background: transparent; height: 0;}}
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {{display: none !important;}}
.block-container {{padding: 0.6rem 1.4rem 3rem; max-width: 1800px;}}
* {{scrollbar-width: thin; scrollbar-color: var(--line2) transparent;}}
h1, h2, h3, h4 {{font-family: var(--mono); letter-spacing: .02em; font-weight: 600; color: var(--text);}}
h3 {{font-size: 1.02rem; text-transform: uppercase; letter-spacing: .09em; color: var(--amber);
     border-bottom: 1px solid var(--line); padding-bottom: 6px; margin: 1.1rem 0 .7rem;}}
h4 {{font-size: .86rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted);}}
p, li, label, span, div {{font-variant-numeric: tabular-nums;}}
[data-testid="stCaptionContainer"], .stCaption, small {{color: var(--muted); font-size: .76rem; line-height: 1.45;}}

/* верхняя командная строка */
.topbar {{display:flex; align-items:stretch; gap:0; border:1px solid var(--line); background:var(--panel); margin:0 0 6px;}}
.topbar .brand {{background: var(--amber); color:#0A0D12; font-weight:800; letter-spacing:.14em; padding:7px 14px; font-size:.86rem; white-space:nowrap;}}
.topbar .crumb {{padding:7px 14px; color:var(--text); font-size:.8rem; letter-spacing:.06em; text-transform:uppercase; border-right:1px solid var(--line); white-space:nowrap;}}
.topbar .crumb b {{color: var(--amber); font-weight:700;}}
.topbar .spacer {{flex:1;}}
.topbar .chip {{padding:7px 14px; font-size:.74rem; letter-spacing:.08em; color:var(--muted); border-left:1px solid var(--line); white-space:nowrap; text-transform:uppercase;}}
.topbar .chip .dot {{display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:7px; vertical-align:middle;}}
.topbar .chip.live {{color: var(--up);}} .topbar .chip.live .dot {{background: var(--up); box-shadow: 0 0 6px var(--up);}}
.topbar .chip.demo {{color: {C['violet']};}} .topbar .chip.demo .dot {{background: {C['violet']};}}

/* бегущая строка */
.tape {{overflow:hidden; white-space:nowrap; background:var(--panel); border:1px solid var(--line); border-top:none; padding:6px 0; margin-bottom:10px; font-size:.78rem;}}
.tape-inner {{display:inline-block; animation: scroll 120s linear infinite;}}
.tape-inner .item {{margin:0 20px; letter-spacing:.03em;}}
.tape-inner .item .k {{color:var(--muted); margin-right:6px; text-transform:uppercase;}}
.tape-inner .item .d {{color:var(--muted); font-size:.68rem; margin-left:6px;}}
@keyframes scroll {{from {{transform:translateX(0);}} to {{transform:translateX(-50%);}}}}

/* карточки показателей */
[data-testid="stMetric"] {{background: var(--panel); border:1px solid var(--line); border-top:2px solid var(--amber); border-radius:0; padding:10px 14px 9px;}}
[data-testid="stMetricLabel"] p {{color:var(--muted); font-size:.68rem; text-transform:uppercase; letter-spacing:.1em; font-weight:500;}}
[data-testid="stMetricValue"] {{font-size:1.5rem; font-weight:600; color:var(--text); letter-spacing:-.01em;}}
[data-testid="stMetricDelta"] {{font-size:.74rem;}}

/* функциональные клавиши: вкладки и горизонтальные переключатели */
.stTabs [data-baseweb="tab-list"] {{gap:2px; border-bottom:1px solid var(--line); flex-wrap: wrap;}}
.stTabs [data-baseweb="tab"] {{height:32px; padding:0 14px; background:var(--panel); border:1px solid var(--line); border-bottom:none; border-radius:0;
    font-size:.72rem; letter-spacing:.1em; font-weight:600; color:var(--muted); text-transform:uppercase;}}
.stTabs [data-baseweb="tab"]:hover {{color:var(--amber); background:var(--panel2);}}
.stTabs [aria-selected="true"] {{background:var(--amber) !important; color:#0A0D12 !important;}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{display:none;}}
div[data-testid="stRadio"] > div[role="radiogroup"] {{gap:4px; flex-wrap:wrap;}}
div[data-testid="stRadio"] [role="radiogroup"] > label {{background:var(--panel); border:1px solid var(--line); padding:4px 12px; margin:0; border-radius:0;
    cursor:pointer; transition:all .12s;}}
div[data-testid="stRadio"] [role="radiogroup"] > label:hover {{border-color:var(--amber-dim); background:var(--panel2);}}
div[data-testid="stRadio"] [role="radiogroup"] > label > div:first-child {{display:none;}}
div[data-testid="stRadio"] [role="radiogroup"] > label p {{font-size:.72rem; letter-spacing:.09em; text-transform:uppercase; color:var(--muted); font-weight:600;}}
div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {{background:var(--amber); border-color:var(--amber);}}
div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p {{color:#0A0D12;}}
[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] {{flex-direction:column;}}
[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] > label {{width:100%;}}

/* кнопки */
.stButton > button, .stDownloadButton > button, [data-testid^="stBaseButton"] {{border-radius:0; border:1px solid var(--line2); background:var(--panel);
    color:var(--text); font-size:.74rem; letter-spacing:.09em; text-transform:uppercase; font-weight:600; padding:.45rem .9rem; transition:all .12s;}}
.stButton > button:hover, .stDownloadButton > button:hover {{border-color:var(--amber); color:var(--amber); background:var(--panel2);}}
[data-testid="stBaseButton-primary"] {{background:var(--amber) !important; color:#0A0D12 !important; border-color:var(--amber) !important;}}
[data-testid="stBaseButton-primary"]:hover {{filter:brightness(1.1);}}

/* поля ввода */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {{border-radius:0 !important; background:var(--panel) !important; border-color:var(--line) !important;}}
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within > div {{border-color:var(--amber) !important;}}
label[data-testid="stWidgetLabel"] p {{font-size:.7rem; letter-spacing:.09em; text-transform:uppercase; color:var(--muted);}}
[data-baseweb="tag"] {{border-radius:0 !important; background:var(--amber-dim) !important;}}

/* панели */
[data-testid="stExpander"] {{border:1px solid var(--line); border-radius:0; background:var(--panel);}}
[data-testid="stExpander"] summary {{font-size:.76rem; letter-spacing:.07em; text-transform:uppercase;}}
[data-testid="stAlert"] {{border-radius:0; border:1px solid var(--line2); border-left:3px solid var(--amber); background:var(--panel);}}
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"], iframe {{border:1px solid var(--line);}}
[data-testid="stPlotlyChart"] {{background:var(--bg);}}
hr {{border-color: var(--line);}}

/* боковая панель */
[data-testid="stSidebar"] {{background:#0C1017; border-right:1px solid var(--line);}}
[data-testid="stSidebar"] .block-container {{padding-top: 1rem;}}
.side-title {{font-weight:800; letter-spacing:.16em; color:var(--amber); font-size:1.05rem; margin-bottom:2px;}}
.side-sub {{color:var(--muted); font-size:.66rem; letter-spacing:.12em; text-transform:uppercase; margin-bottom:12px;}}
.side-sec {{color:var(--muted); font-size:.64rem; letter-spacing:.16em; text-transform:uppercase; border-bottom:1px solid var(--line); margin:14px 0 6px; padding-bottom:3px;}}
.badge {{display:inline-block; padding:2px 10px; font-size:.68rem; font-weight:700; letter-spacing:.08em;}}
.title {{font-size:1.3rem; font-weight:800; letter-spacing:.14em; color:var(--amber);}}
</style>
"""


def inject(st):
    st.markdown(CSS, unsafe_allow_html=True)


def top_bar(mode, page, live, extra=""):
    """Верхняя командная строка: бренд, путь (режим ▸ раздел), статус данных и время UTC."""
    now = datetime.now(timezone.utc)
    status = ("<span class='chip live'><span class='dot'></span>открытые данные · онлайн</span>" if live
              else "<span class='chip demo'><span class='dot'></span>демо · синтетика</span>")
    return (f"<div class='topbar'><div class='brand'>TRADE LAB</div><div class='crumb'>{mode} ▸ <b>{page}</b></div>"
            f"<div class='spacer'></div>{extra}{status}<div class='chip'>{now:%d %b %Y · %H:%M} UTC</div></div>")


def _last_two(path, date_col, val_col):
    d = pd.read_csv(path).dropna(subset=[val_col])
    if len(d) < 2:
        return None
    a, b = d.iloc[-1], d.iloc[-2]
    return float(a[val_col]), float(b[val_col]), str(a[date_col])


def tape_snapshot():
    """Бегущая строка из встроенного снимка: последние значения с датой наблюдения (не «сейчас»)."""
    specs = [("BRENT", "brent.csv.gz", "Date", "Price", "{:,.2f}"), ("GAS HH", "natgas.csv.gz", "Date", "Price", "{:,.2f}"),
             ("GOLD", "gold.csv.gz", "Date", "Price", "{:,.0f}"), ("VIX", "vix.csv.gz", "DATE", "CLOSE", "{:,.2f}"),
             ("US 10Y %", "us10y.csv.gz", "Date", "Rate", "{:,.2f}"), ("S&P 500", "sp500.csv.gz", "Date", "SP500", "{:,.0f}")]
    items = []
    for name, f, dc, vc, fmt in specs:
        try:
            r = _last_two(SNAP / f, dc, vc)
        except Exception:
            r = None
        if not r:
            continue
        v, prev, when = r
        ch = v / prev - 1 if prev else 0.0
        col = C["up"] if ch >= 0 else C["down"]
        items.append(f"<span class='item'><span class='k'>{name}</span><b>{fmt.format(v)}</b> "
                     f"<span style='color:{col}'>{'▲' if ch >= 0 else '▼'}{abs(ch):.2%}</span><span class='d'>{when[:10]}</span></span>")
    try:
        t = pd.read_csv(SNAP / "co2_ppm_annual.csv.gz").sort_values("Year").iloc[-1]
        items.append(f"<span class='item'><span class='k'>CO₂ ppm</span><b>{t['Mean']:.1f}</b><span class='d'>{int(t['Year'])}</span></span>")
    except Exception:
        pass
    if not items:
        return ""
    body = "".join(items)
    return f"<div class='tape'><div class='tape-inner'>{body}{body}</div></div>"
