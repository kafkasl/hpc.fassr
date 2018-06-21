from sqlite3 import Error
from datetime import datetime
from settings.basic import logging

import settings.basic as cfg

import sqlite3

REVENUE = 0

category = ["Revenue USD Mil",
            "Gross Margin %",
            "Operating Income USD Mil",
            "Operating Margin %",
            "Net Income USD Mil",
            "Earnings Per Share USD",
            "Dividends USD",
            "Payout Ratio % *",
            "Shares Mil",
            "Book Value Per Share * USD",
            "Operating Cash Flow USD Mil",
            "Cap Spending USD Mil",
            "Free Cash Flow USD Mil",
            "Free Cash Flow Per Share * USD",
            "Working Capital USD Mil",
            "Revenue",
            "COGS",
            "Gross Margin",
            "SG&A",
            "R&D",
            "Other",
            "Operating Margin",
            "Net Int Inc & Other",
            "EBT Margin",
            "Tax Rate %",
            "Net Margin %",
            "Asset Turnover (Average)",
            "Return on Assets %",
            "Financial Leverage (Average)",
            "Return on Equity %",
            "Return on Invested Capital %",
            "Interest Coverage",
            "Year over Year",
            "3-Year Average",
            "5-Year Average",
            "10-Year Average",
            "Operating Cash Flow Growth % YOY",
            "Free Cash Flow Growth % YOY",
            "Cap Ex as a % of Sales",
            "Free Cash Flow/Sales %",
            "Free Cash Flow/Net Income",
            "Cash & Short-Term Investments",
            "Accounts Receivable",
            "Inventory",
            "Other Current Assets",
            "Total Current Assets",
            "Net PP&E",
            "Intangibles",
            "Other Long-Term Assets",
            "Total Assets"]


def _create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        logging.debug(e)

    return None


def _select(query):
    conn = _create_connection(cfg.FUNDAMENTAL_DB_PATH)

    cur = conn.cursor()
    cur.execute(query)

    rows = cur.fetchall()

    return rows


def _get_symbol_indicator(symbol, indicator):
    """
    Returns revenue in millions of USD
    :param conn: connection to DB
    :return: rows with all
    """

    logging.debug("Querying for %s" % category[indicator])
    query = "SELECT date, value\n" \
            "FROM dow30stocks_keyratios\n" \
            "WHERE `category`='%s' AND symbol='%s'\n" \
            "GROUP BY symbol\n" \
            "ORDER BY date" % (category[indicator], symbol)

    return _select(query)


def get_revenue_per_year(s, years):
    revenues = _get_symbol_indicator(symbol=s, indicator=REVENUE)
    logging.debug("Revenues for symbol %s [%s]:\n" % (s, len(revenues)))
    filtered_revenues = []
    for date, r in revenues:
        logging.debug(" %s: %s" % (date, r))

        do = datetime.strptime(date, "%Y-%m-%d")
        # revenue is reported in million of dollars, we convert it to dollars
        revenue = r * 1e9
        if do.year in years:
            filtered_revenues.append({'date': do, 'revenue': revenue})

    return filtered_revenues


def get_symbols():
    conn = _create_connection(cfg.FUNDAMENTAL_DB_PATH)

    query = "SELECT DISTINCT symbol\n" \
            "FROM dow30stocks_keyratios"
    cur = conn.cursor()
    cur.execute(query)

    return [s[0] for s in cur.fetchall()]
