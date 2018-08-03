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


def graham_screening(symbols: list, datetime_obj: datetime, limit: float):
    """
    1st condition of Graham's criteria: Adequate size of enterprise
    * revenue(x) > 1.5 billion
    :param df: pandas dataframe with stocks
    :param year: current year for screening
    :param limit: minimum revenue for company
    :return: dataframe without the symbols that do not meet the condition
    """
    # revenues = manager.get_revenue_per_year(symbol, (year,))


    # Graham is a yearly indicator so we check previous years data

    year = datetime_obj.year - 1
    for s in symbols:

        stock_price = 0
        try:
            stock_price = s.get_stock_price(datetime_obj)
        except AssertionError:
            logging.warning("Symbol %s has no stock price in %s" % (s, datetime_obj))
            continue
        """
        1st condition of Graham's criteria: Adequate size of enterprise
        * revenue(x) > 1.5 billion
         """
        s.screening['graham']['adequate enterprise size'] = 0
        try:
            if s.get_y('totalrevenue', year) > limit:
                s.screening['graham']['adequate enterprise size'] = 1
        except KeyError:
            logging.warning("Symbol %s has no totalrevenue value for year %s" % (s, year))

        """
        2nd condition of Graham's criteria: Strong financial condition
        * current_assets(x) = 2 * current_liabilities(x)
        * working_capital(x) > 0
        """
        s.screening['graham']['strong financial conditions'] = 0
        try:
            if s.get_y('totalassets', year) > 2 * s.get_y("totalliabilities", year) and \
                            s.get_y('nwc', year) > 0:
                s.screening['graham']['strong financial conditions'] = 1
        except KeyError:
           logging.warning("Symbol %s has no totalliabilities/totalassets value for year %s" % (s,
                                                                                           year))


        """
        3rd condition of Graham's criteria: Earning stability
        * earnings(x, y) > 0, for all y in [currentYear - 10, currentYear]
        """
        years_stable = 0
        for d in range(0, 11):
            try:
                if s.get_y('basiceps', year - d) > 0:
                    years_stable += 1
            except KeyError:
                logging.warning(
                    "Year %s earnings (basiceps) not available for Graham screening" % (year - d))
        print("YEARS STABLE = %s" % years_stable)
        s.screening['graham']['earnings stability'] = (years_stable / 10)

        """
        4th condition of Graham's criteria: Dividend record
        * dividends_payment(x, y) > 0, forall y in [currentYear - 20, currentYear]
        """
        dividends_record = 0
        for d in range(0, 11):
            try:
                if s.get_y('dividendyield', year - d) > 0:
                    dividends_record += 1
            except KeyError:
                logging.warning(
                    "Year %s dividend yield not available for Graham screening" % (year - d))

        s.screening['graham']['dividends record'] = (dividends_record / 20)

        """
        5th condition of Graham's criteria: Earnings growth
        * EPS(x, y) > 1.03 EPS(x, y-1)
        """

        s.screening['graham']['earnings growth'] = 0
        try:
            current_eps = s.get_y('basiceps', year)
            last_year_eps = s.get_y('basiceps', year - 1)
            s.screening['graham']['earnings growth'] = (current_eps / (1.03 * last_year_eps)) - 1
        except KeyError:
            logging.warning("Symbol %s has no basiceps value for year %s or %s" % (s, year,
                                                                                   year-1))


        """
        6th condition of Graham's criteria: Moderate P/E ratio.
        * P/E between 10 and 15 (or for a precise quantity apply Eq. (6.10)).
        """
        s.screening['graham']['moderate p/e ratio'] = 0
        try:
            if 10 < s.get_y('pricetoearnings', year) < 15:
                s.screening['graham']['moderate p/e ratio'] = 1
        except KeyError:
            logging.warning("Symbol %s has no pricetoearnings value for year %s" % (s, year))

        """
        7th condition of Graham's criteria: Moderate price-to-book ratio.
        * The price-to-book ratio should be no more than 1.5.
        """
        s.screening['graham']['moderate price-to-book ratio'] = 0
        try:
            if s.get_y('pricetobook', year) < 1.5:
                s.screening['graham']['moderate price-to-book ratio'] = 1
        except KeyError:
            logging.warning("Symbol %s has no pricetobook value for year %s" % (s, year))

        """
        8th condition of Graham's criteria: Combination of (6) and (7)
        * Pt = sqrt(22.5 * EPS * BVPS)
        * market_price(x) <= Pt
        """
        s.screening['graham']['graham number'] =0
        try:
            pt = sqrt(
                22.5 * s.get_y('basiceps', year) * s.get_y('bookvaluepershare', year))
            s.screening['graham']['graham number'] = pt / stock_price
        except KeyError as e:
            logging.warning("Symbol %s has missing value %s" % (s, e))
        except ValueError:
            logging.warning("Negative Graham number basiceps:%s, bookvaluepershare:%s for symbol "
                            "%s" % (s.get_y('basiceps', year), s.get_y('bookvaluepershare',
                                                                        year), s))

    return symbols


def screen_stocks(manager, tickers: list):
    """
    Filters out the stocks which do not meet all conditions
    revenue(x) > 1.5 billion
    :param manager: data manager to get the data
    :return: list of the stocks that fulfill Graham's criteria
    """

    symbols = manager.build_symbols(tickers=tickers)

    logging.debug("Symbols to be screened: %s" % " ".join([s.id for s in symbols]))

    today = datetime(2018, 8, 1)
    mean, std = {}, {}

    # Compute mean and std of last year indicators to compute z-scores
    for indicator in Symbol.indicators:
        lst_aux = []
        for s in symbols:
            try:
                lst_aux.append(s.get_y(indicator, today.year - 1))
            except KeyError:
                logging.warning("Value for symbol %s, year %s, and indicator %s not found." % (
                    s, today.year-1, indicator))


        print("LST aux: %s" % lst_aux)
        mean[indicator] = np.mean(lst_aux)
        std[indicator] = np.std(lst_aux)

    screened_symbols = graham_screening(symbols=symbols, datetime_obj=today, limit=1.5e9)
    # screened_symbols = z_scores(symbols=symbols, datetime_obj=today, limit=1.5e9)

    from pprint import pformat
    for s in screened_symbols:
        if s.id == 'AAPL':
            logging.debug("\n%s:\n" % (s.id))
            logging.debug(pformat(s.screening))
            logging.debug(pformat(s._data))

    return screened_symbols
