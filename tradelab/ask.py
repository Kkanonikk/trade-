"""Быстрый вопрос: разбор запроса на русском/английском в действие (раздел, показатель, страна). Правила на ключевых словах, без нейросети."""
import re
from difflib import SequenceMatcher

from . import countries, gdata

INTENTS = [
    (("голод", "продовольств", "недоеда", "еда", "food", "hunger"), "ГОЛОД", None),
    (("старик", "старение", "стареет", "пожил", "демограф", "рождаем", "население", "aging", "ageing", "population"), "ДЕМОГРАФИЯ", None),
    (("кризис", "дефолт", "рецесс", "инфляц", "crisis", "recession"), "КРИЗИСЫ", None),
    (("климат", "температур", "co2", "выброс", "потепл", "уровень моря", "ледник", "climate"), "КЛИМАТ", None),
    (("торговл", "экспорт", "импорт", "товарооборот", "trade"), "ТОРГОВЛЯ", None),
    (("танкер", "пролив", "судов", "порт", "ship"), "МОРЕ", None),
    (("конфликт", "война", "войн", "боев", "war"), "КОНФЛИКТЫ", None),
    (("валют", "курс", "fx"), "ВАЛЮТЫ", None),
    (("сравни", "сопостав", "compare", "vs"), "СРАВНЕНИЕ", None),
]


def _norm(t):
    return re.sub(r"[^\w\s]", " ", t.lower().replace("ё", "е"))


def find_country(text):
    t = _norm(text)
    best = None
    for iso in countries.table().index:
        for name in (countries.short_ru(iso), countries.table().loc[iso, "name_en"]):
            n = _norm(str(name)).strip()
            if len(n) >= 4 and (n in t or (len(n) >= 5 and n[:-1] in t)):
                if best is None or len(n) > best[1]:
                    best = (iso, len(n))
    return best[0] if best else None


STOP = {"где", "что", "как", "все", "для", "покажи", "показать", "больше", "всего", "меньше", "страна", "страны", "стран",
        "будет", "риск", "есть", "это", "при", "над", "про", "мира", "мире", "какие", "какая", "какой", "сколько"}


def find_indicators(text, avail, top=3):
    """Показатели с названием, наиболее похожим на слова запроса."""
    words = [w for w in _norm(text).split() if len(w) >= 3 and w not in STOP]
    scored = []
    for _, r in avail.iterrows():
        label = _norm(r["label"])
        hits = sum(1.0 for w in words if w[:5] in label)
        score = hits + max((SequenceMatcher(None, w, label).ratio() for w in words), default=0) * 0.3
        if hits >= 1:
            scored.append((score, r["key"]))
    return [k for _, k in sorted(scored, reverse=True)[:top]]


def interpret(text, avail):
    """Возвращает dict: page, iso, indicators, explanation."""
    t = _norm(text)
    iso = find_country(text)
    inds = find_indicators(text, avail)
    page = next((p for kws, p, _ in INTENTS if any(k in t for k in kws)), None)
    if iso and page is None and not inds:
        page = "СТРАНА"
    if page is None and inds:
        page = "ИССЛЕДОВАТЕЛЬ"
    if page is None:
        return {"page": None, "iso": iso, "indicators": inds, "explanation": "Не понял запрос. Попробуй: «где больше всего стариков», «риск голода», «кризис в Турции»."}
    meta = gdata.meta()
    expl = f"Открываю раздел {page}" + (f", страна {countries.short_ru(iso)}" if iso else "") + (f", показатель «{meta[inds[0]]['label']}»" if inds and page in ("ИССЛЕДОВАТЕЛЬ", "СТРАНА") else "")
    return {"page": page, "iso": iso, "indicators": inds, "explanation": expl}
