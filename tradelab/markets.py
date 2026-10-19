"""Справочник мировых рынков, валют, макро-рядов и торговых часов."""
from datetime import datetime, time
from zoneinfo import ZoneInfo

# name -> (iso3, yahoo, stooq, город, lat, lon, tz, открытие, закрытие)
MARKETS = {
    "S&P 500": ("USA", "^GSPC", "^spx", "Нью-Йорк", 40.71, -74.00, "America/New_York", (9, 30), (16, 0)),
    "Nasdaq": ("USA", "^IXIC", "^ndq", "Нью-Йорк (Nasdaq)", 41.6, -72.8, "America/New_York", (9, 30), (16, 0)),
    "TSX (Канада)": ("CAN", "^GSPTSE", None, "Торонто", 43.65, -79.38, "America/Toronto", (9, 30), (16, 0)),
    "Bovespa": ("BRA", "^BVSP", None, "Сан-Паулу", -23.55, -46.63, "America/Sao_Paulo", (10, 0), (17, 55)),
    "IPC (Мексика)": ("MEX", "^MXX", None, "Мехико", 19.43, -99.13, "America/Mexico_City", (8, 30), (15, 0)),
    "FTSE 100": ("GBR", "^FTSE", "^ukx", "Лондон", 51.50, -0.12, "Europe/London", (8, 0), (16, 30)),
    "DAX": ("DEU", "^GDAXI", "^dax", "Франкфурт", 50.11, 8.68, "Europe/Berlin", (9, 0), (17, 30)),
    "CAC 40": ("FRA", "^FCHI", "^cac", "Париж", 48.86, 2.35, "Europe/Paris", (9, 0), (17, 30)),
    "SMI (Швейцария)": ("CHE", "^SSMI", None, "Цюрих", 47.37, 8.54, "Europe/Zurich", (9, 0), (17, 30)),
    "IMOEX": ("RUS", "IMOEX.ME", None, "Москва", 55.75, 37.62, "Europe/Moscow", (10, 0), (18, 50)),
    "BIST 100": ("TUR", "XU100.IS", None, "Стамбул", 41.01, 28.97, "Europe/Istanbul", (10, 0), (18, 0)),
    "JSE (ЮАР)": ("ZAF", "^J203.JO", None, "Йоханнесбург", -26.20, 28.04, "Africa/Johannesburg", (9, 0), (17, 0)),
    "Tadawul": ("SAU", "^TASI.SR", None, "Эр-Рияд", 24.71, 46.68, "Asia/Riyadh", (10, 0), (15, 0)),
    "Sensex": ("IND", "^BSESN", None, "Мумбаи", 19.08, 72.88, "Asia/Kolkata", (9, 15), (15, 30)),
    "Shanghai": ("CHN", "000001.SS", "^shc", "Шанхай", 31.23, 121.47, "Asia/Shanghai", (9, 30), (15, 0)),
    "Hang Seng": ("HKG", "^HSI", "^hsi", "Гонконг", 22.32, 114.17, "Asia/Hong_Kong", (9, 30), (16, 0)),
    "Nikkei 225": ("JPN", "^N225", "^nkx", "Токио", 35.68, 139.69, "Asia/Tokyo", (9, 0), (15, 30)),
    "KOSPI": ("KOR", "^KS11", None, "Сеул", 37.57, 126.98, "Asia/Seoul", (9, 0), (15, 30)),
    "STI (Сингапур)": ("SGP", "^STI", None, "Сингапур", 1.35, 103.82, "Asia/Singapore", (9, 0), (17, 0)),
    "ASX 200": ("AUS", "^AXJO", None, "Сидней", -33.87, 151.21, "Australia/Sydney", (10, 0), (16, 0)),
}

# Сырьё, риск-индикаторы, крипта (Yahoo)
CROSS_ASSETS = {
    "Нефть Brent": "BZ=F", "Нефть WTI": "CL=F", "Золото": "GC=F", "Серебро": "SI=F", "Медь": "HG=F",
    "Газ (Henry Hub)": "NG=F", "Индекс доллара": "DX-Y.NYB", "VIX": "^VIX", "Биткоин": "BTC-USD",
}

# Валюты: ЦБ РФ (код ЦБ, номинал учитывается при разборе) и ECB через Frankfurter
FX_CODES_CBR = {"USD": "R01235", "EUR": "R01239", "CNY": "R01375", "GBP": "R01035", "JPY": "R01820", "CHF": "R01775"}
FX_ECB = ["EUR", "GBP", "JPY", "CHF", "CNY", "CAD", "AUD", "TRY", "INR", "BRL", "KRW", "MXN", "ZAR", "SGD"]

# Ключевые ставки центробанков: страна -> (название ЦБ, ряд FRED, ISO3). Ряды FRED перечислены по памяти:
# недоступные пропускаются и видны в статусе источников.
POLICY_RATES = {
    "США (ФРС)": ("DFF", "USA"), "Еврозона (ЕЦБ)": ("ECBMRRFR", "DEU"),
    "Великобритания": ("IRSTCI01GBM156N", "GBR"), "Япония": ("IRSTCI01JPM156N", "JPN"),
    "Канада": ("IRSTCI01CAM156N", "CAN"), "Австралия": ("IRSTCI01AUM156N", "AUS"),
    "Швейцария": ("IRSTCI01CHM156N", "CHE"), "Южная Корея": ("IRSTCI01KRM156N", "KOR"),
    "Мексика": ("IRSTCI01MXM156N", "MEX"), "Китай": ("INTDSRCNM193N", "CHN"),
    "Индия": ("INTDSRINM193N", "IND"), "Бразилия": ("INTDSRBRM193N", "BRA"),
    "Россия": ("INTDSRRUM193N", "RUS"), "Турция": ("INTDSRTRM193N", "TUR"), "ЮАР": ("INTDSRZAM193N", "ZAF"),
}
YIELDS = {"US 2Y": "DGS2", "US 10Y": "DGS10", "US 3M": "DGS3MO", "Германия 10Y": "IRLTLT01DEM156N"}

WB_INDICATORS = {"Рост ВВП, %": "NY.GDP.MKTP.KD.ZG", "Инфляция, %": "FP.CPI.TOTL.ZG",
                 "Безработица, %": "SL.UEM.TOTL.ZS"}
WB_COUNTRIES = sorted({m[0] for m in MARKETS.values()} | {"RUS", "SAU", "IDN", "IND", "CHN"})


def market_status(name, now_utc=None):
    """Открыта ли биржа сейчас (по будням, без учёта праздников). Возвращает (открыта, местное время)."""
    iso, _, _, _, _, _, tz, op, cl = MARKETS[name]
    now = (now_utc or datetime.now(ZoneInfo("UTC"))).astimezone(ZoneInfo(tz))
    is_open = now.weekday() < 5 and time(*op) <= now.time() <= time(*cl)
    return is_open, now
