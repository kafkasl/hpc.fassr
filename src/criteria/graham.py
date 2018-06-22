# Graham criteria for stock screening
from math import sqrt
from settings.basic import logging

import settings.basic as cfg
# import datetime as dt
import pandas as pd


def _adequate_size_of_enterprise(df, year, limit):
    """
    1st condition of Graham's criteria: Adequate size of enterprise
    * revenue(x) > 1.5 billion
    :param df: pandas dataframe with stocks
    :param year: current year for screening
    :param limit: minimum revenue for company
    :return: dataframe without the symbols that do not meet the condition
    """
    # revenues = manager.get_revenue_per_year(symbol, (year,))

    # TODO check that all stocks in DOW 30 pass these filter (as expected?)
    symbols = set(df.loc[(df['category'] == 'Revenue USD Mil') &
                         (df['value'] >= limit) &
                         (df['date'].dt.year == year), 'symbol'])

    # Filter symbols that do not meet the conditions
    df = df[df['symbol'].isin(symbols)]

    return df


def _strong_financial_conditions(df, year):
    """
    2nd condition of Graham's criteria: Strong financial condition
    * current_assets(x) = 2 * current_liabilities(x)
    * working_capital(x) > 0
    :param df: pandas dataframe with stocks
    :param year: current year for screening
    :return: dataframe without the symbols that do not meet the condition
    """

    assets = df.loc[(df['category'] == 'Total Current Assets') &
                    (df['date'].dt.year == year), ['symbol', 'value', 'date']] \
        .rename(columns={'value': 'assets'})

    liabilities = df.loc[(df['category'] == 'Total Current Liabilities') &
                         (df['date'].dt.year == year), ['symbol', 'value', 'date']] \
        .rename(columns={'value': 'liabilities'})

    aux = pd.merge(assets, liabilities, on=['symbol', 'date'])
    symbols = aux.loc[aux['assets'] > 2 * aux['liabilities'], 'symbol']

    # Filter symbols that do not meet the conditions
    df = df.loc[df['symbol'].isin(symbols)]

    return df


def _earnings_stability(df, year):
    """
    3rd condition of Graham's criteria: Earning stability
    * earnings(x, y) > 0, for all y in [currentYear - 10, currentYear]
    :param df: pandas dataframe with stocks
    :param year: current year for screening
    :return: dataframe without the symbols that do not meet the condition
    """
    aux = df.loc[(df['category'] == 'Earnings Per Share USD') &
                 (df['date'].dt.year > year - 10) &
                 (df['date'].dt.year <= year), ['symbol', 'date', 'value']]

    symbols = []
    for s in set(df['symbol']):
        # for a given symbol, we order it by increasing year, select the values, and check if they are
        # monotonically increasing
        if aux.loc[aux['symbol'] == s].sort_values('date', ascending=True)['value'].is_monotonic_increasing:
            symbols.append(s)

    # Filter symbols that do not meet the conditions
    df = df.loc[df['symbol'].isin(symbols)]

    return df


def _dividends_record(df, year):
    """
    4th condition of Graham's criteria: Dividend record
    * dividents_payment(x, y) > 0, forall y in [currentYear - 20, currentYear]
    :param df: pandas dataframe with stocks
    :param year: current year for screening
    :return: dataframe without the symbols that do not meet the condition
    """
    aux = df.loc[(df['category'] == 'Dividends USD') &
                 (df['date'].dt.year > year - 10) &
                 (df['date'].dt.year <= year) &
                 (df['value'] > 0), 'symbol']

    symbols = []
    for s in set(aux):
        # for a given symbol, if it appears 10 times, it means that its dividends were greater than 0 in the past
        # 10 years (because of the previous date filtering)
        if len(aux.loc[aux == s]) == 10:
            symbols.append(s)

    # Filter symbols that do not meet the conditions
    df = df.loc[df['symbol'].isin(symbols)]

    return df


# TODO: SHOULD CHECK ONLY LAST YEAR?
def _earnings_growth(df, year):
    """
    5th condition of Graham's criteria: Earnings growth
    * EPS(x, y) > 1.03 EPS(x, y-1)
    :param df: pandas dataframe with stocks
    :param year: current year for screening
    :return: dataframe without the symbols that do not meet the condition
    """
    # Filter symbols that do not meet the conditions
    df = df.loc[df['symbol'].isin(symbols)]

    return df


def _graham_number(df, year):
    """
    8th condition of Graham's criteria: Combination of (6) and (7)
    * Pt = sqrt(22.5 * EPS * BVPS)
    * market_price(x) <= Pt
    :param df: pandas dataframe with stocks
    :param year: current year for screening
    :return: dataframe without the symbols that do not meet the condition
    """
    pt = sqrt(22.5 * stock['eps'][0] * stock['bvps'][0])
    # Filter symbols that do not meet the conditions
    df = df.loc[df['symbol'].isin(symbols)]

    return df


def screen_stocks(manager):
    """
    Filters out the stocks which do not meet all conditions
    revenue(x) > 1.5 billion
    :param stocks: list of stocks in json format complying with
    :return: list of the stocks that fulfill Graham's criteria
    """
    df = manager.get_fundamental_df()
    symbols = manager.get_symbols(df)

    logging.debug("Symbols to be screened:")
    [logging.debug(s) for s in symbols]

    df = _adequate_size_of_enterprise(df=df, year=cfg.GRAHAM['year'], limit=cfg.GRAHAM['revenue_limit'])
    df = _strong_financial_conditions(df=df, year=cfg.GRAHAM['year'])
    df = _earnings_stability(df=df, year=cfg.GRAHAM['year'])
    df = _dividends_record(df=df, year=cfg.GRAHAM['year'])
    df = _earnings_growth(df=df, year=cfg.GRAHAM['year'])
    df = _graham_number(df=df, year=cfg.GRAHAM['year'])
    # symbols = [s for s in symbols if _strong_financial_conditions(s)]
    # symbols = [s for s in symbols if _earnings_stability(s)]
    # symbols = [s for s in symbols if _dividends_record(s)]
    # symbols = [s for s in symbols if _earnings_growth(s)]
    # symbols = [s for s in symbols if _graham_number(s)]

    return symbols
