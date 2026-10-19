"""Комплаенс: лимиты на портфель, проверка до сделки (целевые веса) и после (текущие позиции)."""
from dataclasses import dataclass

from . import config, risk


@dataclass
class Limits:
    max_position: float = 0.25     # макс. доля одной бумаги
    max_sector: float = 0.50       # макс. доля одного сектора
    min_cash: float = 0.0          # мин. доля денег
    max_var95: float = 0.05        # макс. дневная VaR 95% (доля капитала)
    max_turnover: float = 1.0      # макс. оборот за ребаланс (доля капитала, сумма |Δw|)


def check(weights, limits, var95=None, current=None, sector_of=config.sector):
    """Возвращает список нарушений: [{'Правило', 'Факт', 'Лимит', 'Описание'}]. Пусто = всё в порядке."""
    breaches = []
    invested = sum(weights.values())

    def add(rule, fact, limit, text):
        breaches.append({"Правило": rule, "Факт": fact, "Лимит": limit, "Описание": text})

    for t, w in weights.items():
        if w > limits.max_position + 1e-9:
            add("Доля бумаги", w, limits.max_position, f"{t} превышает лимит позиции")
    for sec, w in risk.sector_exposure(weights, sector_of).items():
        if w > limits.max_sector + 1e-9:
            add("Доля сектора", w, limits.max_sector, f"Сектор «{sec}» превышает лимит")
    cash = 1.0 - invested
    if cash < limits.min_cash - 1e-9:
        add("Мин. деньги", cash, limits.min_cash, "Недостаточно свободных денег")
    if var95 is not None and var95 > limits.max_var95 + 1e-9:
        add("VaR 95% (1д)", var95, limits.max_var95, "Дневной риск выше лимита")
    if current is not None:
        names = set(weights) | set(current)
        turnover = sum(abs(weights.get(t, 0.0) - current.get(t, 0.0)) for t in names)
        if turnover > limits.max_turnover + 1e-9:
            add("Оборот", turnover, limits.max_turnover, "Ребаланс слишком велик")
    return breaches
