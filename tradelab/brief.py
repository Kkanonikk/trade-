"""Автоматическое страновое досье: короткие выводы из данных с источниками и честными оговорками; экспорт в HTML (печать в PDF из браузера)."""
import html
from datetime import date

import numpy as np
import plotly.graph_objects as go

from . import aging, countries, forecast, gdata, hunger, registry

CAVEAT = ("Выводы построены автоматически по открытым данным. Прогнозы ООН показаны как прогнозы, статистические прогнозы приложения "
          "это экстраполяция трендов, а индексы уязвимости описывают сравнительный риск, а не предсказывают события.")


def _rank(w, key, iso, year):
    xs = gdata.cross_section(w, key, year)
    if iso not in xs.index:
        return None, None
    return int(xs.rank(ascending=False, method="min")[iso]), len(xs)


def _last(w, key, iso):
    s = gdata.series(w, key, iso)
    cap = gdata.last_actual_year(key)
    s = s[s.index <= cap] if cap else s
    return (int(s.index[-1]), float(s.iloc[-1])) if len(s) else (None, None)


def build(w, iso, pop_path):
    """Возвращает dict: title, bullets [(текст, [ключи источников])], figures [(подпись, plotly figure)], warnings."""
    name = countries.short_ru(iso)
    bullets, figs = [], []
    # население и возраст
    if iso in pop_path.index:
        p = pop_path.loc[iso]
        now, p50, p100 = p.get(2025), p.get(2050), p.get(2100)
        if np.isfinite(now) and np.isfinite(p50):
            peak = int(p.loc[2024:2100].idxmax())
            txt = (f"Население {now / 1e6:,.1f} млн (2025); к 2050 по прогнозу ООН {p50 / 1e6:,.1f} млн ({p50 / now - 1:+.0%}), "
                   f"к 2100 {p100 / 1e6:,.1f} млн. " + ("Пик населения уже пройден или близок." if peak <= 2030 else f"Пик ожидается около {peak} года."))
            bullets.append((txt, [aging.POPULATION, aging.GROWTH_POP]))
        fig = go.Figure(go.Scatter(x=p.dropna().index, y=p.dropna().values / 1e6, line=dict(color="#FF9F1A")))
        fig.add_vline(x=2024.5, line_dash="dot")
        fig.update_layout(title="Население, млн (после 2024 прогноз ООН)", template="plotly_white", height=320)
        figs.append(("Население", fig))
    age = gdata.series(w, aging.MEDIAN_AGE, iso)
    if len(age) and 2025 in age.index and 2050 in age.index:
        r, n = _rank(w, aging.MEDIAN_AGE, iso, 2025)
        bullets.append((f"Медианный возраст {age[2025]:.1f} лет (место {r} из {n} по возрасту), к 2050 по прогнозу ООН {age[2050]:.1f} "
                        f"({age[2050] - age[2025]:+.1f} лет).", [aging.MEDIAN_AGE]))
    # экономика и климат
    for key, fmt in (("gm:gdppercapita_us_inflation_adjusted", "ВВП на душу {v:,.0f} USD ({y})"),
                     ("gm:life_expectancy_at_birth_data_from_ihme", "Ожидаемая продолжительность жизни {v:.1f} лет ({y})"),
                     ("co2:co2_per_capita", "Выбросы CO₂ {v:.1f} т на человека ({y})"),
                     ("en:renewables_share_elec", "Доля возобновляемых в электроэнергии {v:.0f}% ({y})")):
        y, v = _last(w, key, iso)
        if y is not None:
            r, n = _rank(w, key, iso, y)
            bullets.append((fmt.format(v=v, y=y) + (f", место {r} из {n}" if r else ""), [key]))
    # голод
    idx, contrib = hunger.index(w, 2022)
    if iso in idx.index:
        top = contrib.loc[iso].sort_values(ascending=False).head(3)
        why = ", ".join(hunger.COMPONENTS[k][2].lower() for k in top.index)
        r = int(idx.rank(ascending=False)[iso])
        bullets.append((f"Продовольственная уязвимость {idx[iso]:.0f} из 100 (место {r} из {len(idx)}, чем выше, тем уязвимее). "
                        f"Главные факторы: {why}.", list(hunger.COMPONENTS)))
        row = contrib.loc[iso].dropna().sort_values()
        fig = go.Figure(go.Bar(x=row.values, y=[hunger.COMPONENTS[k][2] for k in row.index], orientation="h", marker_color="#FF4D4F"))
        fig.update_layout(title="Из чего складывается уязвимость, баллов", template="plotly_white", height=360)
        figs.append(("Уязвимость", fig))
    return {"title": f"Досье: {name}", "iso": iso, "bullets": bullets, "figures": figs, "caveat": CAVEAT}


def to_html(brief, forecast_note=""):
    """Самодостаточный HTML-отчёт: выводы, графики (plotly.js встроен), перечень источников с лицензиями."""
    ids = []
    for _, keys in brief["bullets"]:
        for k in keys:
            s = gdata.meta().get(k, {}).get("source")
            if s and s not in ids:
                ids.append(s)
    parts = [f"<!doctype html><meta charset='utf-8'><title>{html.escape(brief['title'])}</title>"
             "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:30px auto;padding:0 16px;color:#1a1f29}"
             "h1{border-bottom:3px solid #FF9F1A}li{margin:8px 0}.cav{background:#fff6e5;padding:10px;border-left:4px solid #FF9F1A}"
             "small{color:#555}</style>",
             f"<h1>{html.escape(brief['title'])}</h1><small>Сформировано {date.today():%d.%m.%Y} приложением Trade Lab</small><h2>Главное</h2><ul>"]
    parts += [f"<li>{html.escape(t)}</li>" for t, _ in brief["bullets"]]
    parts.append("</ul>")
    if forecast_note:
        parts.append(f"<p>{html.escape(forecast_note)}</p>")
    for i, (cap, fig) in enumerate(brief["figures"]):
        parts.append(fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False))
    parts.append("<h2>Источники</h2><ul>")
    for i in ids:
        d = registry.SOURCES.get(i)
        if d:
            parts.append(f"<li><b>{html.escape(d['name'])}</b>, {html.escape(d['org'])}: {html.escape(d['url'])} · лицензия: {html.escape(d['license'])}</li>")
    parts.append(f"</ul><p class='cav'>{html.escape(brief['caveat'])}</p>")
    return "\n".join(parts)
