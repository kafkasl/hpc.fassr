from sqlite3 import Error
from datetime import datetime
from settings.basic import logging

import settings.basic as cfg
import pandas as pd

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


def get_fundamental_df():
    conn = _create_connection(cfg.FUNDAMENTAL_DB_PATH)
    data = pd.read_sql('SELECT * FROM dow30stocks_keyratios', conn)

    # Set date as datetime field
    data['date'] = pd.to_datetime(data['date'])

    return data


def get_symbols(data):
    return set(data['symbol'])