"""Реестр показателей и источников: единое место, откуда берутся подписи, единицы, группы и ссылки на источники.

Каждый показатель имеет ключ вида 'gm:имя' (Gapminder), 'co2:колонка' (OWID CO2), 'en:колонка' (OWID Energy),
'wp:имя' (панель WEO/World Bank из снимка или живых данных). Везде в интерфейсе показывается источник из этого реестра.
"""

# id источника -> метаданные
SOURCES = {
    "gapminder": {"name": "Gapminder Systema Globalis", "org": "Gapminder Foundation (агрегирует ООН, ВОЗ, Всемирный банк, IHME, FAO и др.)",
                  "url": "https://github.com/open-numbers/ddf--gapminder--systema_globalis", "license": "CC BY 4.0",
                  "note": "Прогнозы населения, медианного возраста и иждивенчества до 2100 г. это средний вариант ООН (World Population Prospects)."},
    "un_wpp": {"name": "UN World Population Prospects (через Gapminder)", "org": "ООН, Отдел народонаселения",
               "url": "https://population.un.org/wpp/", "license": "CC BY 3.0 IGO",
               "note": "Оценки до 2023 г., дальше проекция среднего варианта; не прогноз событий."},
    "owid_co2": {"name": "OWID CO₂ and Greenhouse Gas Emissions", "org": "Our World in Data (на основе Global Carbon Project, Climate Watch, BP, EIA)",
                 "url": "https://github.com/owid/co2-data", "license": "CC BY 4.0", "note": ""},
    "owid_energy": {"name": "OWID Energy", "org": "Our World in Data (на основе Energy Institute, Ember, EIA)",
                    "url": "https://github.com/owid/energy-data", "license": "CC BY 4.0", "note": ""},
    "imf_weo": {"name": "IMF World Economic Outlook", "org": "Международный валютный фонд", "url": "https://www.imf.org/en/Publications/WEO",
                "license": "условия использования IMF (открытый доступ)", "note": "В снимке факты до 2013 г. (зеркало datahub.io), дальше проекции МВФ в обучение не берутся."},
    "worldbank": {"name": "World Bank Open Data (WDI)", "org": "Всемирный банк", "url": "https://data.worldbank.org", "license": "CC BY 4.0",
                  "note": "В снимке: ВВП, инфляция, население; остальное через живую загрузку."},
    "who": {"name": "WHO Global Health Observatory", "org": "ВОЗ", "url": "https://www.who.int/data/gho", "license": "CC BY-NC-SA 3.0 IGO", "note": "Через API, живая загрузка."},
    "ucdp": {"name": "UCDP Georeferenced Event Dataset", "org": "Uppsala Conflict Data Program", "url": "https://ucdp.uu.se", "license": "CC BY 4.0",
             "note": "События конфликтов; загрузка без ключа."},
    "portwatch": {"name": "IMF PortWatch", "org": "МВФ и Оксфордский университет", "url": "https://portwatch.imf.org", "license": "CC BY 4.0", "note": "Оценки по AIS."},
    "natural_earth": {"name": "Natural Earth", "org": "NACIS", "url": "https://www.naturalearthdata.com", "license": "общественное достояние", "note": "Границы стран."},
    "fed_h10": {"name": "Federal Reserve H.10 (зеркало datahub.io)", "org": "ФРС США", "url": "https://www.federalreserve.gov/releases/h10/",
                "license": "общественное достояние", "note": ""},
    "temp": {"name": "Global Temperature (NASA GISS GISTEMP и NOAA GCAG, зеркало datahub.io)", "org": "NASA GISS, NOAA NCEI",
             "url": "https://github.com/datasets/global-temp", "license": "общественное достояние (PDDL)", "note": "Аномалия относительно 1951–1980."},
    "noaa_gml": {"name": "NOAA Global Monitoring Laboratory: CO₂ Мауна-Лоа (зеркало datahub.io)", "org": "NOAA GML",
                 "url": "https://gml.noaa.gov/ccgg/trends/", "license": "общественное достояние", "note": ""},
    "sea_level": {"name": "Sea Level Rise (EPA на основе CSIRO и NOAA)", "org": "EPA США, CSIRO, NOAA",
                  "url": "https://github.com/datasets/sea-level-rise", "license": "общественное достояние (PDDL)", "note": ""},
    "wgms": {"name": "Glacier Mass Balance (WGMS через datahub.io)", "org": "World Glacier Monitoring Service",
             "url": "https://github.com/datasets/glacier-mass-balance", "license": "открытые данные", "note": ""},
    "shiller": {"name": "Shiller U.S. Stock Markets 1871–present", "org": "Robert Shiller (Yale)", "url": "http://www.econ.yale.edu/~shiller/data.htm",
                "license": "открытый доступ", "note": "Зеркало datahub.io."},
}

# Показатели Gapminder: имя -> (подпись, единица, группа, "лучше": up/down/none, источник)
G = "gapminder"
GAPMINDER = {
    # --- демография
    "total_population_with_projections": ("Население", "чел.", "Демография", "none", "un_wpp"),
    "male_population_with_projections": ("Мужчины", "чел.", "Демография", "none", "un_wpp"),
    "female_population_with_projections": ("Женщины", "чел.", "Демография", "none", "un_wpp"),
    "population_growth_annual_percent_with_projections": ("Прирост населения", "% в год", "Демография", "none", "un_wpp"),
    "median_age_years": ("Медианный возраст", "лет", "Демография", "none", "un_wpp"),
    "children_and_elderly_per_100_adults": ("Иждивенцы (дети и пожилые) на 100 взрослых", "чел.", "Демография", "down", "un_wpp"),
    "crude_birth_rate_births_per_1000_population": ("Рождаемость", "на 1000 чел.", "Демография", "none", G),
    "crude_death_rate_deaths_per_1000_population": ("Смертность", "на 1000 чел.", "Демография", "down", G),
    "surviving_kids_per_woman": ("Выживающих детей на женщину", "чел.", "Демография", "none", G),
    "dead_kids_per_woman": ("Умерших детей на женщину", "чел.", "Демография", "down", G),
    "population_density_per_square_km": ("Плотность населения", "чел./км²", "Демография", "none", G),
    "urban_population_percent_of_total": ("Городское население", "% населения", "Демография", "none", G),
    "sex_ratio_all_age_groups": ("Соотношение полов (мужчин на 100 женщин)", "", "Демография", "none", G),
    "teen_fertility_rate_births_per_1000_women_ages_15_19": ("Подростковая рождаемость", "на 1000 женщин 15–19 лет", "Демография", "down", G),
    "age_at_1st_marriage_women": ("Возраст первого брака у женщин", "лет", "Демография", "none", G),
    "contraceptive_use_percent_of_women_ages_15_49": ("Контрацепция среди женщин 15–49", "%", "Демография", "none", G),
    "new_births_total_number_estimated": ("Число рождений", "чел. в год", "Демография", "none", G),
    # --- здоровье
    "life_expectancy_at_birth_data_from_ihme": ("Ожидаемая продолжительность жизни (IHME)", "лет", "Здоровье", "up", G),
    "life_expectancy_female": ("Продолжительность жизни, женщины", "лет", "Здоровье", "up", G),
    "life_expectancy_male": ("Продолжительность жизни, мужчины", "лет", "Здоровье", "up", G),
    "infant_mortality_rate_per_1000_births": ("Младенческая смертность", "на 1000 родившихся", "Здоровье", "down", G),
    "under_five_mortality_from_cme_per_1000_born": ("Смертность до 5 лет", "на 1000 родившихся", "Здоровье", "down", G),
    "maternal_mortality_ratio_who": ("Материнская смертность (ВОЗ)", "на 100 тыс. родов", "Здоровье", "down", G),
    "births_attended_by_skilled_health_staff_percent_of_total": ("Роды с квалифицированной помощью", "%", "Здоровье", "up", G),
    "total_health_spending_percent_of_gdp": ("Расходы на здравоохранение", "% ВВП", "Здоровье", "up", G),
    "government_health_spending_per_person_us": ("Госрасходы на здравоохранение на душу", "USD", "Здоровье", "up", G),
    "medical_doctors_per_1000_people": ("Врачей на 1000 человек", "чел.", "Здоровье", "up", G),
    "adults_with_hiv_percent_age_15_49": ("ВИЧ среди взрослых 15–49", "%", "Здоровье", "down", G),
    "all_forms_of_tb_incidence_per_100000_estimated": ("Заболеваемость туберкулёзом", "на 100 тыс.", "Здоровье", "down", G),
    "malaria_deaths_per_100000_reported": ("Смертность от малярии", "на 100 тыс.", "Здоровье", "down", G),
    "smoking_adults_percent_of_population_over_age_15": ("Курящие взрослые", "%", "Здоровье", "down", G),
    "alcohol_consumption_per_adult_15plus_litres": ("Потребление алкоголя на взрослого", "литров чистого спирта", "Здоровье", "down", G),
    "body_mass_index_bmi_men_kgperm2": ("Индекс массы тела, мужчины", "кг/м²", "Здоровье", "none", G),
    "body_mass_index_bmi_women_kgperm2": ("Индекс массы тела, женщины", "кг/м²", "Здоровье", "none", G),
    "suicide_per_100000_people": ("Самоубийства", "на 100 тыс.", "Здоровье", "down", G),
    "epidemic_deaths_annual_number": ("Погибшие в эпидемиях", "чел. в год", "Здоровье", "down", G),
    "epidemic_affected_annual_number": ("Затронутые эпидемиями", "чел. в год", "Здоровье", "down", G),
    "at_least_basic_water_source_overall_access_percent": ("Доступ к чистой воде", "% населения", "Здоровье", "up", G),
    "at_least_basic_sanitation_overall_access_percent": ("Доступ к санитарии", "% населения", "Здоровье", "up", G),
    # --- питание и голод
    "food_supply_kilocalories_per_person_and_day": ("Доступные калории на человека", "ккал в сутки", "Питание и голод", "up", G),
    "malnutrition_weight_for_age_percent_of_children_under_5": ("Недостаточный вес детей до 5 лет", "% детей", "Питание и голод", "down", G),
    "underweight_children": ("Дети с недостаточным весом", "% детей", "Питание и голод", "down", G),
    "agricultural_land_percent_of_land_area": ("Сельхозземли", "% площади", "Питание и голод", "none", G),
    "agriculture_percent_of_gdp": ("Сельское хозяйство", "% ВВП", "Питание и голод", "none", G),
    "agriculture_workers_percent_of_employment": ("Занятые в сельском хозяйстве", "% занятых", "Питание и голод", "none", G),
    "internal_renewable_water_cu_meters_per_person": ("Возобновляемые водные ресурсы на человека", "м³", "Питание и голод", "up", G),
    "extreme_poverty_percent_people_below_300_a_day": ("Крайняя бедность (< 3 USD в день)", "% населения", "Питание и голод", "down", G),
    "poverty_percent_people_below_830_a_day": ("Бедность (< 8.3 USD в день)", "% населения", "Питание и голод", "down", G),
    "inequality_index_gini": ("Неравенство (индекс Джини)", "0–100", "Питание и голод", "down", G),
    "aid_received_percent_of_gni": ("Полученная помощь", "% ВНД", "Питание и голод", "none", G),
    # --- экономика
    "income_per_person_with_projections": ("ВВП на душу (ППС, с прогнозом)", "USD", "Экономика", "up", G),
    "gdppercapita_us_inflation_adjusted": ("ВВП на душу (реальный)", "USD 2015", "Экономика", "up", G),
    "total_gdp_us_inflation_adjusted": ("ВВП (реальный)", "USD 2015", "Экономика", "up", G),
    "gdp_total_yearly_growth": ("Рост ВВП", "% в год", "Экономика", "up", G),
    "gdp_per_capita_yearly_growth": ("Рост ВВП на душу", "% в год", "Экономика", "up", G),
    "inflation_annual_percent": ("Инфляция", "% в год", "Экономика", "down", G),
    "exports_percent_of_gdp": ("Экспорт", "% ВВП", "Экономика", "none", G),
    "imports_percent_of_gdp": ("Импорт", "% ВВП", "Экономика", "none", G),
    "trade_balance_percent_of_gdp": ("Торговый баланс", "% ВВП", "Экономика", "up", G),
    "investments_percent_of_gdp": ("Инвестиции", "% ВВП", "Экономика", "up", G),
    "tax_revenue_percent_of_gdp": ("Налоговые доходы", "% ВВП", "Экономика", "none", G),
    "debt_to_foreigners_by_public_and_private_percent_of_gni": ("Внешний долг", "% ВНД", "Экономика", "down", G),
    "total_reserves_percent_of_debt_to_foreigners": ("Резервы к внешнему долгу", "%", "Экономика", "up", G),
    "foreign_direct_investment_net_inflows_percent_of_gdp": ("Прямые инвестиции (приток)", "% ВВП", "Экономика", "up", G),
    "central_bank_discount_rate_annual_percent": ("Учётная ставка ЦБ", "%", "Экономика", "none", G),
    "industry_percent_of_gdp": ("Промышленность", "% ВВП", "Экономика", "none", G),
    "services_percent_of_gdp": ("Услуги", "% ВВП", "Экономика", "none", G),
    "high_technology_exports_percent_of_manufactured_exports": ("Высокотехнологичный экспорт", "% экспорта промтоваров", "Экономика", "up", G),
    "market_value_of_listed_companies_percent_of_gdp": ("Капитализация компаний", "% ВВП", "Экономика", "none", G),
    "hdi_human_development_index": ("Индекс человеческого развития", "0–1", "Экономика", "up", G),
    "gdpperworking_hour_us_inflation_adjusted": ("Производительность труда (ВВП за час)", "USD 2017", "Экономика", "up", G),
    # --- труд
    "aged_15plus_unemployment_rate_percent": ("Безработица", "% рабочей силы", "Труд", "down", G),
    "aged_15plus_labour_force_participation_rate_percent": ("Участие в рабочей силе (15+)", "%", "Труд", "up", G),
    "aged_65plus_labour_force_participation_rate_percent": ("Участие в рабочей силе (65+)", "%", "Труд", "none", G),
    "long_term_unemployment_rate_percent": ("Долгосрочная безработица", "%", "Труд", "down", G),
    "working_hours_per_week": ("Рабочие часы в неделю", "часов", "Труд", "none", G),
    # --- бедствия и климатические риски
    "drought_affected_annual_number": ("Пострадавшие от засухи", "чел. в год", "Бедствия", "down", G),
    "drought_deaths_annual_number": ("Погибшие от засухи", "чел. в год", "Бедствия", "down", G),
    "flood_affected_annual_number": ("Пострадавшие от наводнений", "чел. в год", "Бедствия", "down", G),
    "flood_deaths_annual_number": ("Погибшие от наводнений", "чел. в год", "Бедствия", "down", G),
    "extreme_temperature_affected_annual_number": ("Пострадавшие от экстремальной температуры", "чел. в год", "Бедствия", "down", G),
    "extreme_temperature_deaths_annual_number": ("Погибшие от экстремальной температуры", "чел. в год", "Бедствия", "down", G),
    "storm_affected_annual_number": ("Пострадавшие от штормов", "чел. в год", "Бедствия", "down", G),
    "storm_deaths_annual_number": ("Погибшие от штормов", "чел. в год", "Бедствия", "down", G),
    "earthquake_deaths_annual_number": ("Погибшие от землетрясений", "чел. в год", "Бедствия", "down", G),
    # --- энергия и среда
    "forest_coverage_percent": ("Лесистость", "% площади", "Климат и энергия", "up", G),
    "electricity_generation_per_person": ("Выработка электроэнергии на душу", "кВт·ч", "Климат и энергия", "up", G),
    "energy_use_per_person": ("Потребление энергии на душу", "кг нефтяного эквивалента", "Климат и энергия", "none", G),
    "oil_production_total": ("Добыча нефти", "т в год", "Климат и энергия", "none", G),
    "natural_gas_production_total": ("Добыча газа", "м³ в год", "Климат и энергия", "none", G),
    "co2_intensity_of_economic_output_kg_co2_per_2021_ppp_of_gdp": ("Углеродоёмкость экономики", "кг CO₂ на USD ВВП", "Климат и энергия", "down", G),
    "pump_price_for_gasoline_us_per_liter": ("Цена бензина", "USD за литр", "Климат и энергия", "none", G),
    # --- технологии
    "internet_users": ("Пользователи интернета", "% населения", "Технологии", "up", G),
    "cell_phones_per_100_people": ("Мобильные телефоны на 100 человек", "шт.", "Технологии", "up", G),
    "broadband_subscribers_per_100_people": ("Широкополосный интернет на 100 человек", "шт.", "Технологии", "up", G),
    "personal_computers_per_100_people": ("Компьютеры на 100 человек", "шт.", "Технологии", "up", G),
    "patent_applications_total": ("Патентные заявки", "шт. в год", "Технологии", "up", G),
    # --- безопасность и институты
    "democracy_score_use_as_color": ("Индекс демократии", "баллы", "Безопасность и институты", "up", G),
    "military_expenditure_percent_of_gdp": ("Военные расходы", "% ВВП", "Безопасность и институты", "none", G),
    "battle_deaths_per_100_000": ("Погибшие в боях", "на 100 тыс.", "Безопасность и институты", "down", G),
    "murder_per_100000_people": ("Убийства", "на 100 тыс.", "Безопасность и институты", "down", G),
    # --- образование
    "literacy_rate_adult_total_percent_of_people_ages_15_and_above": ("Грамотность взрослых", "%", "Образование", "up", G),
    "primary_completion_rate_total_percent_of_relevant_age_group": ("Окончание начальной школы", "%", "Образование", "up", G),
    "mean_years_in_school_women_of_reproductive_age_15_to_44": ("Годы обучения женщин 15–44", "лет", "Образование", "up", G),
}

# OWID CO2: колонка -> (подпись, единица, группа, лучше)
OWID_CO2 = {
    "co2": ("Выбросы CO₂", "млн т", "Климат и энергия", "down"),
    "co2_per_capita": ("Выбросы CO₂ на душу", "т на человека", "Климат и энергия", "down"),
    "co2_per_gdp": ("Выбросы CO₂ на единицу ВВП", "кг на USD", "Климат и энергия", "down"),
    "cumulative_co2": ("Накопленные выбросы CO₂", "млн т", "Климат и энергия", "none"),
    "share_global_co2": ("Доля в мировых выбросах CO₂", "%", "Климат и энергия", "none"),
    "methane": ("Метан", "млн т CO₂-экв.", "Климат и энергия", "down"),
    "nitrous_oxide": ("Закись азота", "млн т CO₂-экв.", "Климат и энергия", "down"),
    "total_ghg": ("Парниковые газы, всего", "млн т CO₂-экв.", "Климат и энергия", "down"),
    "ghg_per_capita": ("Парниковые газы на душу", "т CO₂-экв.", "Климат и энергия", "down"),
    "temperature_change_from_ghg": ("Вклад в потепление (парниковые газы)", "°C", "Климат и энергия", "down"),
    "coal_co2": ("CO₂ от угля", "млн т", "Климат и энергия", "down"),
    "oil_co2": ("CO₂ от нефти", "млн т", "Климат и энергия", "down"),
    "gas_co2": ("CO₂ от газа", "млн т", "Климат и энергия", "down"),
    "land_use_change_co2": ("CO₂ от землепользования", "млн т", "Климат и энергия", "down"),
    "consumption_co2_per_capita": ("Потребительский след CO₂ на душу", "т на человека", "Климат и энергия", "down"),
    "trade_co2": ("CO₂ во внешней торговле (нетто-импорт)", "млн т", "Климат и энергия", "none"),
    "energy_per_gdp": ("Энергоёмкость ВВП", "кВт·ч на USD", "Климат и энергия", "down"),
}
OWID_ENERGY = {
    "primary_energy_consumption": ("Потребление первичной энергии", "ТВт·ч", "Климат и энергия", "none"),
    "energy_per_capita": ("Энергопотребление на душу", "кВт·ч", "Климат и энергия", "none"),
    "renewables_share_energy": ("Доля возобновляемых в энергии", "%", "Климат и энергия", "up"),
    "fossil_share_energy": ("Доля ископаемого топлива в энергии", "%", "Климат и энергия", "down"),
    "low_carbon_share_energy": ("Доля низкоуглеродной энергии", "%", "Климат и энергия", "up"),
    "electricity_generation": ("Выработка электроэнергии", "ТВт·ч", "Климат и энергия", "none"),
    "renewables_share_elec": ("Доля возобновляемых в электроэнергии", "%", "Климат и энергия", "up"),
    "fossil_share_elec": ("Доля ископаемого в электроэнергии", "%", "Климат и энергия", "down"),
    "carbon_intensity_elec": ("Углеродоёмкость электроэнергии", "г CO₂/кВт·ч", "Климат и энергия", "down"),
    "solar_electricity": ("Солнечная электроэнергия", "ТВт·ч", "Климат и энергия", "up"),
    "wind_electricity": ("Ветровая электроэнергия", "ТВт·ч", "Климат и энергия", "up"),
    "oil_production": ("Добыча нефти", "ТВт·ч", "Климат и энергия", "none"),
    "gas_production": ("Добыча газа", "ТВт·ч", "Климат и энергия", "none"),
    "coal_production": ("Добыча угля", "ТВт·ч", "Климат и энергия", "none"),
    "net_elec_imports_share_demand": ("Чистый импорт электроэнергии", "% спроса", "Климат и энергия", "none"),
}

# Первые года прогнозов (для показателей «с прогнозами»): до этого года данные это оценки
PROJECTION_FROM = {n: 2024 for n in (
    "total_population_with_projections", "male_population_with_projections", "female_population_with_projections",
    "population_growth_annual_percent_with_projections", "median_age_years", "children_and_elderly_per_100_adults",
    "surviving_kids_per_woman", "dead_kids_per_woman", "life_expectancy_female", "life_expectancy_male",
    "crude_death_rate_deaths_per_1000_population", "sex_ratio_all_age_groups", "population_density_per_square_km")}
PROJECTION_FROM["income_per_person_with_projections"] = 2019


def key_gm(name):
    return f"gm:{name}"


def all_meta():
    """Плоский словарь: ключ показателя -> {label, unit, group, better, source}."""
    out = {}
    for n, (lab, unit, grp, better, src) in GAPMINDER.items():
        out[key_gm(n)] = {"label": lab, "unit": unit, "group": grp, "better": better, "source": src,
                          "projection_from": PROJECTION_FROM.get(n)}
    for n, (lab, unit, grp, better) in OWID_CO2.items():
        out[f"co2:{n}"] = {"label": lab, "unit": unit, "group": grp, "better": better, "source": "owid_co2", "projection_from": None}
    for n, (lab, unit, grp, better) in OWID_ENERGY.items():
        out[f"en:{n}"] = {"label": lab, "unit": unit, "group": grp, "better": better, "source": "owid_energy", "projection_from": None}
    return out


def cite(keys):
    """Строка-источник для подписи под графиком: названия и ссылки уникальных источников."""
    meta = all_meta()
    ids = []
    for k in keys:
        s = meta.get(k, {}).get("source") or ("imf_weo" if k.startswith("wp:") else None)
        if s and s not in ids:
            ids.append(s)
    return "Источник: " + "; ".join(f"{SOURCES[i]['name']} ({SOURCES[i]['url']}, {SOURCES[i]['license']})" for i in ids if i in SOURCES)
