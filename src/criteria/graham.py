# Graham criteria for stock screening
from math import sqrt
from settings.basic import logging

import settings.basic as cfg



# TODO define a format file for the default json stock value
def _adequate_size_of_enterprise(symbol, manager):
    """
    1st condition of Graham's criteria: Adequate size of enterprise:
    * revenue(x) > 1.5 billion
    :param stock: stock in json format complying with
    :return: true if it passes the condition
    """
    year = cfg.CURRENT_YEAR
    limit = int(1.5e9)
    revenues = manager.get_revenue_per_year(symbol, (year,))

    if len(revenues) != 1:
        logging.warning("Discarding %s because it has %s revenue entries for year %s" % (symbol, len(revenues), year))
        return False

    assert revenues[0]['date'].year == year

    return revenues[0]['revenue'] > limit


def _strong_financial_conditions(symbol, manager):
    """
    2nd condition of Graham's criteria: Strong financial condition:
    * current_assets(x) = 2 * current_liabilities(x)
    * working_capital(x) > 0
    :param stock: stock in json format complying with
    :return: true if it passes the condition
    """
    return (stock['assets'][0] >= stock['liabilities'][0]) and stock['working_capital'] > 0


def _earnings_stability(symbol, manager):
    """
    3rd condition of Graham's criteria: Earning stability:
    * earnings(x, y) > 0, forall y in [currentYear - 10, currentYear]
    :param stock: stock in json format complying with
    :return: true if it passes the condition
    """
    for y in range(0, 10):
        if not stock['earnings'][y] > 0:
            return False
    return True


def _dividends_record(symbol, manager):
    """
    4th condition of Graham's criteria:    Adequate size of enterprise:
    revenue(x) > 1.5 billion
    :param stock: stock in json format complying with
    :return: true if it passes the condition
    """
    for y in range(0, 20):
        if not stock['dividends_payment'][y] > 0:
            return False
    return True


# TODO: SHOULD CHECK ONLY LAST YEAR?
def _earnings_growth(symbol, manager):
    """
    5th condition of Graham's criteria:
    Adequate size of enterprise:
    revenue(x) > 1.5 billion
    :param stock: stock in json format complying with
    :return: true if it passes the condition
    """
    return stock['eps'][0] > 1.03 * stock['eps'][1]


def _graham_number(symbol, manager):
    """
    8th condition of Graham's criteria:
    Adequate size of enterprise:
    revenue(x) > 1.5 billion
    :param stock: stock in json format complying with
    :return: true if it passes the condition
    """
    pt = sqrt(22.5 * stock['eps'][0] * stock['bvps'][0])
    return stock['market_price'] <= pt


def screen_stocks(manager):
    """
    Filters out the stocks which do not meet all conditions
    revenue(x) > 1.5 billion
    :param stocks: list of stocks in json format complying with
    :return: list of the stocks that fulfill Graham's criteria
    """
    symbols = manager.get_symbols()
    logging.debug("Symbols to be screened:")
    [logging.debug(s) for s in symbols]
    symbols = [s for s in symbols if _adequate_size_of_enterprise(s, manager)]
    # symbols = [s for s in symbols if _strong_financial_conditions(s)]
    # symbols = [s for s in symbols if _earnings_stability(s)]
    # symbols = [s for s in symbols if _dividends_record(s)]
    # symbols = [s for s in symbols if _earnings_growth(s)]
    # symbols = [s for s in symbols if _graham_number(s)]

    return symbols
