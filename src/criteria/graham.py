# Graham criteria for stock screening
from math import sqrt
from functools import reduce
from settings.basic import logging

import settings.basic as cfg
import pandas as pd


def _filter(df, symbols):
    # Filter symbols that do not meet the conditions
    df = df[df['symbol'].isin(symbols)]

    return df


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
    symbols = list(df.loc[(df['category'] == 'Revenue USD Mil') &
                          (df['value'] >= limit) &
                          (df['date'].dt.year == year), 'symbol'].drop_duplicates())

    return symbols


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
    symbols = list(aux.loc[aux['assets'] > 2 * aux['liabilities'], 'symbol'].drop_duplicates())

    return symbols


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
        if aux.loc[aux['symbol'] == s].sort_values('date', ascending=True)[
            'value'].is_monotonic_increasing:
            symbols.append(s)

    return symbols


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

    return symbols


# TODO: SHOULD CHECK ONLY LAST YEAR?
def _earnings_growth(df, year):
    """
    5th condition of Graham's criteria: Earnings growth
    * EPS(x, y) > 1.03 EPS(x, y-1)
    :param df: pandas dataframe with stocks
    :param year: current year for screening
    :return: dataframe without the symbols that do not meet the condition
    """

    current_eps = df.loc[(df['category'] == 'Earnings Per Share USD') &
                         (df['date'].dt.year == year), ['symbol', 'value']] \
        .rename(columns={'value': 'current_eps'})

    last_year_eps = df.loc[(df['category'] == 'Earnings Per Share USD') &
                           (df['date'].dt.year == year - 1), ['symbol', 'value']] \
        .rename(columns={'value': 'last_year_eps'})

    aux = pd.merge(current_eps, last_year_eps, on=['symbol'])

    symbols = list(
        aux.loc[aux['current_eps'] > 1.03 * aux['last_year_eps'], 'symbol'].drop_duplicates())

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
                 (df['date'].dt.year == year), ['symbol', 'value']] \
        .rename(columns={'value': 'eps'})

    bvps = df.loc[(df['category'] == 'Book Value Per Share * USD') &
                  (df['date'].dt.year == year), ['symbol', 'value']] \
        .rename(columns={'value': 'bvps'})

    per = df.loc[(df['category'] == 'Price to Earnings') &
                 (df['date'].dt.year == year), ['symbol', 'value']] \
        .rename(columns={'value': 'per'})

    shares_number = df.loc[(df['category'] == 'Shares Mil') &
                           (df['date'].dt.year == year), ['symbol', 'value']] \
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

    logging.debug("Symbols to be screened:")
    [logging.debug(s.id) for s in symbols]

    screened_symbols = []
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
