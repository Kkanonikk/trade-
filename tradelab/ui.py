"""Общие элементы оформления интерфейса."""
import numpy as np

C = dict(up="#26C281", down="#FF4D4F", accent="#FF9F1A", blue="#4DA3FF", violet="#B38CFF",
         grid="#1E2735", text="#C9D1DC", bg="#0A0D12", panel="#131923")
PALETTE = [C["accent"], C["blue"], C["violet"], C["up"], C["down"], "#56B6C2", "#E5C07B", "#98C379",
           "#C678DD", "#61AFEF", "#E06C75", "#D19A66"]


def style(fig, height=450):
    fig.update_layout(template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
                      font=dict(family="monospace", color=C["text"]), height=height,
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(gridcolor=C["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False)
    return fig


def pct(x, d=1):
    return "н/д" if x is None or not np.isfinite(x) else f"{x:.{d}%}"
