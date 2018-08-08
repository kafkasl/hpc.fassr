# Graham criteria for stock screening
from dateutil.relativedelta import relativedelta as delta

from datetime import datetime
from functools import reduce
from math import sqrt

from data_managers.symbol import Symbol
from settings.basic import logging

import settings.basic as cfg
import pandas as pd
import numpy as np


def _filter(df, symbols):
    # Filter symbols that do not meet the conditions
    df = df[df['symbol'].isin(symbols)]

    return df


def _new_graham_screening(id_: str, name: str, date: datetime):
    info = {
        "symbol": id_,
        "name": name,
        "type": "Graham Criteria",
        "date": date.strftime(cfg.DATE_FORMAT),
        "indicators": [
            {
                "name": "adequate enterprise size",
                "description": "1st condition of Graham's criteria: Adequate size of enterprise. Revenue(x) > 1.5 billion",
                "value": 0,
                "exposure": 1

            }, {
                "name": "strong financial conditions",
                "description": "2nd condition of Graham's criteria: Strong financial condition. Current_assets(x) = 2 * current_liabilities(x) * working_capital(x) > 0",
                "value": 0,
                "exposure": 1

            }, {
                "name": "earnings stability",
                "description": "3rd condition of Graham's criteria: Earning stability. Earnings(x, y) > 0, for all y in [currentYear - 10, currentYear]",
                "value": 0,
                "exposure": 1

            }, {
                "name": "dividends record",
                "description": "4th condition of Graham's criteria: Dividend record.  Dividends_payment(x, y) > 0, forall y in [currentYear - 20, currentYear]",
                "value": 0,
                "exposure": 1

            }, {
                "name": "earnings growth",
                "description": "5th condition of Graham's criteria: Earnings growth. EPS(x, "
                               "y) > 1.03 EPS(x, y-1)",
                "value": 0,
                "exposure": 1

            }, {
                "name": "moderate p/e ratio",
                "description": "6th condition of Graham's criteria: Moderate P/E ratio. P/E between 10 and 15 (or for a precise quantity apply Eq. (6.10)).",
                "value": 0,
                "exposure": 1

            }, {
                "name": "moderate price-to-book ratio",
                "description": "7th condition of Graham's criteria: Moderate price-to-book ratio. The price-to-book ratio should be no more than 1.5.",
                "value": 0,
                "exposure": 1

            }, {
                "name": "graham number",
                "description": "    8th condition of Graham's criteria: Combination of (6) and (7). Pt = sqrt(22.5 * EPS * BVPS). Market_price(x) <= Pt",
                "value": 0,
                "exposure": 1

            }
        ]
    }
    return info


def graham_screening(symbol: Symbol, datetime_obj: datetime, limit: float):
    """
    1st condition of Graham's criteria: Adequate size of enterprise
    * revenue(x) > 1.5 billion
    :param symbol: string representing stock symbol
    :param datetime_obj: date of the screening
    :param limit: minimum revenue for company
    :return: dataframe without the symbols that do not meet the condition
    """
    # revenues = manager.get_revenue_per_year(symbol, (year,))


    # Graham is a yearly indicator so we check previous years data

    screening_info = _new_graham_screening(symbol.id, symbol.name, datetime_obj)

    year = datetime_obj.year - 1

    stock_price = 0
    try:
        stock_price = symbol.get_stock_price(datetime_obj)
    except AssertionError:
        logging.warning("Symbol %s has no stock price in %s" % (symbol, datetime_obj))
        raise AssertionError("Symbol %s has no stock price in %s" % (symbol, datetime_obj))

    # 1st condition of Graham's criteria: Adequate size of enterprise
    try:
        if symbol.get_y('totalrevenue', year) > limit:
            screening_info['indicators'][0]['value'] = 1
    except KeyError:
        logging.warning("Symbol %s has no totalrevenue value for year %s" % (symbol, year))

    # 2nd condition of Graham's criteria: Strong financial condition
    try:
        if symbol.get_y('totalassets', year) > 2 * symbol.get_y("totalliabilities", year) and \
                        symbol.get_y('nwc', year) > 0:
            screening_info['indicators'][1]['value'] = 1
    except KeyError:
        logging.warning(
            "Symbol %s has no totalliabilities/totalassets value for year %s" % (symbol,
                                                                                 year))

    # 3rd condition of Graham's criteria: Earning stability
    years_stable = 0
    for d in range(0, 11):
        try:
            if symbol.get_y('basiceps', year - d) > 0:
                years_stable += 1
        except KeyError:
            logging.warning(
                "Year %s earnings (basiceps) not available for Graham screening" % (year - d))
    print("YEARS STABLE = %s" % years_stable)
    screening_info['indicators'][2]['value'] = (years_stable / 10)

   # 4th condition of Graham's criteria: Dividend record
    dividends_record = 0
    for d in range(0, 11):
        try:
            if symbol.get_y('dividendyield', year - d) > 0:
                dividends_record += 1
        except KeyError:
            logging.warning(
                "Year %s dividend yield not available for Graham screening" % (year - d))

        screening_info['indicators'][3]['value'] = (dividends_record / 20)

    # 5th condition of Graham's criteria: Earnings growth
    try:
        current_eps = symbol.get_y('basiceps', year)
        last_year_eps = symbol.get_y('basiceps', year - 1)
        screening_info['indicators'][4]['value'] = (current_eps / (1.03 * last_year_eps)) - 1
    except KeyError:
        logging.warning("Symbol %s has no basiceps value for year %s or %s" % (symbol, year,
                                                                               year - 1))

    # 6th condition of Graham's criteria: Moderate P/E ratio.
    try:
        if 10 < symbol.get_y('pricetoearnings', year) < 15:
            screening_info['indicators'][5]['value'] = 1
    except KeyError:
        logging.warning("Symbol %s has no pricetoearnings value for year %s" % (symbol, year))

    # 7th condition of Graham's criteria: Moderate price-to-book ratio.
    try:
        if symbol.get_y('pricetobook', year) < 1.5:
            screening_info['indicators'][6]['value'] = 1
    except KeyError:
        logging.warning("Symbol %s has no pricetobook value for year %s" % (symbol, year))

    # 8th condition of Graham's criteria: Combination of (6) and (7)
    try:
        pt = sqrt(
            22.5 * symbol.get_y('basiceps', year) * symbol.get_y('bookvaluepershare', year))
        screening_info['indicators'][7]['value'] = pt / stock_price
    except KeyError as e:
        logging.warning("Symbol %s has missing value %s" % (symbol, e))
    except ValueError:
        logging.warning("Negative Graham number basiceps:%s, bookvaluepershare:%s for symbol "
                        "%s" % (symbol.get_y('basiceps', year), symbol.get_y('bookvaluepershare',
                                                                             year), symbol))

    return screening_info


def screen_stocks(manager, tickers: list):
    """
    Filters out the stocks which do not meet all conditions
    revenue(x) > 1.5 billion
    :param manager: data manager to get the data
    :return: list of the stocks that fulfill Graham's criteria
    """

    symbols = manager.build_symbols(tickers=tickers)

    logging.debug("Symbols to be screened: %s" % " ".join([id_ for id_ in symbols.keys()]))

    today = datetime(2018, 8, 1)
    mean, std = {}, {}

    # Compute mean and std of last year indicators to compute z-scores
    for indicator in Symbol.indicators:
        lst_aux = []
        for s in symbols.values():
            try:
                lst_aux.append(s.get_y(indicator, today.year - 1))
            except KeyError:
                logging.warning("Value for symbol %s, year %s, and indicator %s not found." % (
                    s, today.year - 1, indicator))

        print("LST aux: %s" % lst_aux)
        mean[indicator] = np.mean(lst_aux)
        std[indicator] = np.std(lst_aux)

    screened_symbols = {}
    for id, symbol in symbols.items():
        screening_info = graham_screening(symbol=symbol, datetime_obj=today, limit=1.5e9)
        if screening_info:
            screened_symbols[id] = screening_info
    # screened_symbols = z_scores(symbols=symbols, datetime_obj=today, limit=1.5e9)

    from pprint import pformat
    logging.debug("\n%s:\n" % ("AAPL"))
    logging.debug(pformat(screened_symbols['AAPL']))
    logging.debug(pformat(symbols['AAPL']._data))

    return screened_symbols
