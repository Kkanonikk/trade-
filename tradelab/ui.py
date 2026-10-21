"""Общие элементы оформления интерфейса."""
import numpy as np

C = dict(up="#26C281", down="#FF4D4F", accent="#FF9F1A", blue="#4DA3FF", violet="#B38CFF",
         grid="#1A2230", text="#D7DEE8", bg="#0A0D12", panel="#10151D")
PALETTE = [C["accent"], C["blue"], C["violet"], C["up"], C["down"], "#56B6C2", "#E5C07B", "#98C379",
           "#C678DD", "#61AFEF", "#E06C75", "#D19A66"]


MONO = "JetBrains Mono, IBM Plex Mono, SF Mono, Consolas, monospace"


def style(fig, height=450):
    """Единый вид графиков: чёрный фон, тонкая сетка, янтарная палитра, подсказки в стиле терминала."""
    fig.update_layout(template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], colorway=PALETTE,
                      font=dict(family=MONO, color=C["text"], size=11), height=height,
                      margin=dict(l=12, r=12, t=34, b=12), legend=dict(orientation="h", y=1.1, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor="#10151D", bordercolor=C["accent"], font=dict(family=MONO, size=11, color=C["text"])),
                      title=dict(font=dict(size=12, color="#9AA5B5")), hovermode=fig.layout.hovermode or "closest")
    fig.update_xaxes(gridcolor=C["grid"], gridwidth=0.5, zeroline=False, linecolor="#2E3A4E", ticks="outside", tickcolor="#2E3A4E",
                     tickfont=dict(size=10), title_font=dict(size=10, color="#8B96A5"))
    fig.update_yaxes(gridcolor=C["grid"], gridwidth=0.5, zeroline=False, linecolor="#2E3A4E", ticks="outside", tickcolor="#2E3A4E",
                     tickfont=dict(size=10), title_font=dict(size=10, color="#8B96A5"))
    return fig


def pct(x, d=1):
    return "н/д" if x is None or not np.isfinite(x) else f"{x:.{d}%}"
