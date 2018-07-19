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


def graham_screening(symbols: list, d: datetime, limit: float):
    """
    1st condition of Graham's criteria: Adequate size of enterprise
    * revenue(x) > 1.5 billion
    :param df: pandas dataframe with stocks
    :param year: current year for screening
    :param limit: minimum revenue for company
    :return: dataframe without the symbols that do not meet the condition
    """
    # revenues = manager.get_revenue_per_year(symbol, (year,))


    for s in symbols:
        """
        1st condition of Graham's criteria: Adequate size of enterprise
        * revenue(x) > 1.5 billion
        """
        s.screening['graham']['adequate size of enterprise'] = \
            (s.get_y('totalrevenue', d) / limit) - 1

        """
        2nd condition of Graham's criteria: Strong financial condition
        * current_assets(x) = 2 * current_liabilities(x)
        * working_capital(x) > 0
        """
        s.screening['graham']['strong financial conditions'] = 0
        if s.get_y('totalassets', d) >\
                        2 * s.get_y("totalliabilities", d) and \
                        s.get_y('nwc', d) > 0:
            s.screening['graham']['strong financial conditions'] = 1

        """
        3rd condition of Graham's criteria: Earning stability
        * earnings(x, y) > 0, for all y in [currentYear - 10, currentYear]
        """
        years_stable = [s.get_y('eps', d-delta(years=d)) > 0 for d in range(0, 11)]\
            .count(True)
        s.screening['graham']['earnings stability'] = (years_stable / 10) - 1

        """
        4th condition of Graham's criteria: Dividend record
        * dividends_payment(x, y) > 0, forall y in [currentYear - 20, currentYear]
        """
        dividends_paid = [s.get_y('dividendyield', d - delta(years=d)) > 0 for d in range(0, 11)]\
            .count(True)
        s.screening['graham']['dividends record'] = (dividends_paid / 20) - 1

        """
        5th condition of Graham's criteria: Earnings growth
        * EPS(x, y) > 1.03 EPS(x, y-1)
        """
        current_eps = s.get_y('basiceps', d)
        last_year_eps = s.get_y('basiceps', d - delta(years=1))
        s.screening['graham']['earnings growth'] = (current_eps/ (1.03 * last_year_eps)) - 1

        """
        6th condition of Graham's criteria: Moderate P/E ratio.
        * P/E between 10 and 15 (or for a precise quantity apply Eq. (6.10)).
        """
        s.screening['graham']['moderate p/e ratio'] = 0
        if 10 < s.get_y('pricetoearnings', d) < 15:
            s.screening['graham']['moderate p/e ratio'] = 1

        """
        7th condition of Graham's criteria: Moderate price-to-book ratio.
        * The price-to-book ratio should be no more than 1.5.
        """
        s.screening['graham']['moderate price-to-book ratio'] = 0
        if s.get_y('pricetobook', d) < 1.5:
            s.screening['graham']['moderate price-to-book ratio'] = 1

        """
        8th condition of Graham's criteria: Combination of (6) and (7)
        * Pt = sqrt(22.5 * EPS * BVPS)
        * market_price(x) <= Pt
        """
        pt = sqrt(22.5 * s.get_y('basiceps', d)* s.get_y('bookvaluepershare', d))
        s.screening['graham']['graham number'] = (pt / s.get_stock_price(d)) - 1

    return symbols



def _graham_number(df, year):
    """
    8th condition of Graham's criteria: Combination of (6) and (7)
    * Pt = sqrt(22.5 * EPS * BVPS)
    * market_price(x) <= Pt
    :param df: pandas dataframe with stocks
    :param year: current year for screening
    :return: dataframe without the symbols that do not meet the condition
    """

    eps = df.loc[(df['category'] == 'Earnings Per Share USD') &
                 (df['d'].dt.year == year), ['symbol', 'value']] \
        .rename(columns={'value': 'eps'})

    bvps = df.loc[(df['category'] == 'Book Value Per Share * USD') &
                  (df['d'].dt.year == year), ['symbol', 'value']] \
        .rename(columns={'value': 'bvps'})

    per = df.loc[(df['category'] == 'Price to Earnings') &
                 (df['d'].dt.year == year), ['symbol', 'value']] \
        .rename(columns={'value': 'per'})

    shares_number = df.loc[(df['category'] == 'Shares Mil') &
                           (df['d'].dt.year == year), ['symbol', 'value']] \
        .rename(columns={'value': 'shares_number'})

    dfs = [eps, bvps, per, shares_number]
    aux = reduce(lambda left, right: pd.merge(left, right, on='symbol'), dfs)

    aux['pt'] = aux.apply(lambda x: sqrt(22.5 * x.eps * x.bvps), axis=1)
    # TODO check if this is a valid way of computing the market price (number of shares is in millions)
    # TODO this is wrong
    aux['market_price'] = aux.apply(lambda x: x.per * x.eps, axis=1)

    symbols = list(aux.loc[aux['market_price'] < aux['pt'], 'symbol'].drop_duplicates())

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

    today = datetime.today() - delta(years=1)
    mean, std = {}, {}

    for indicator in Symbol.indicators:
        lst_aux = [s.get_y(indicator, today) for s in symbols]
        print("LST aux: %s" % lst_aux)
        mean[indicator] = np.mean(lst_aux)
        std[indicator] = np.std(lst_aux)

    screened_symbols = graham_screening(symbols=symbols, d=today, limit=1.5e9)

    for s in screened_symbols:
        print("\n%s:\n%s" % (s.id, s.screening))
    #
    # symbols = [None] * 6
    # symbols[0] = _adequate_size_of_enterprise(df=df, year=cfg.GRAHAM['year'],
    #                                           limit=cfg.GRAHAM['revenue_limit'])
    # symbols[1] = _strong_financial_conditions(df=df, year=cfg.GRAHAM['year'])
    # symbols[2] = _earnings_stability(df=df, year=cfg.GRAHAM['year'])
    # symbols[3] = _dividends_record(df=df, year=cfg.GRAHAM['year'])
    # symbols[4] = _earnings_growth(df=df, year=cfg.GRAHAM['year'])
    # symbols[5] = _graham_number(df=df, year=cfg.GRAHAM['year'])
    #
    # for i, s in enumerate(symbols):
    #     logging.info("Symbols passing condition %s: \n\t%s\n" % (i + 1, s))
    #
    # # Flatten list to check which symbols pass all conditions
    # screened_symbols = set(symbols[0]).intersection(*symbols)

    return screened_symbols
