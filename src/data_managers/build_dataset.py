import pickle
import itertools

import numpy as np

from utils.cache import call_and_cache
from string import Template
from collections import OrderedDict
from collections import defaultdict
from dateutil.relativedelta import relativedelta

"""
The goal of this file is to download a fundamental dataset with the following features:
* Contains all possible reportable attributes of income statement, cash flow, balance sheet.
* N/A will be coded as -1 (after checking that no -1 value exists).
* Check if calculations / current offer also interesting values.

"""


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


fundamentals_url = Template("https://api.intrinio.com/financials/standardized?"
                            "identifier=${symbol}&"
                            "statement=${statement}&"
                            "type=TTM&"
                            "fiscal_year=${year}"
                            "period=${period}")

start_year = 2007
end_year = 2018
year_range = list(range(start_year, end_year))
periods = ["Q1TTM", "Q2TTM", "Q3TTM", "FY"]
statements = ["income_statement", "balance_sheet", "cash_flow_statement"]

period2index = {'Q1TTM': 'Q1', 'Q2TTM': 'Q2', 'Q3TTM': 'Q3', 'FY': 'Q4'}
# period2index = {'Q1TTM': 'Q1', 'Q2TTM': 'Q2', 'Q3TTM': 'Q3', 'FY': 'Q4'}

period_keys = ['%s-%s' % (y, q) for y, q in itertools.product(year_range, period2index.values())]

period2index

# TODO: remember to update dow30 with ALL symbols ever to be there to avoid survivor bias

symbols = open('../data/dow30_symbols.lst').read().split()
# symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT']

series_financials_dict = {}
series_periods = set()
for symbol in symbols:
    period_dict = {}
    for year in year_range:
        for period in periods:

            symbol_dict = {}

            for statement in statements:
                url = fundamentals_url.substitute(symbol=symbol, statement=statement, year=year,
                                                  period=period)

                data_json = call_and_cache(url)

                statement_dict = {}
                for element in data_json['data']:
                    statement_dict[element['tag']] = element['value']

                symbol_dict.update(statement_dict)

            period_key = "%s-%s" % (year, period2index[period])
            series_periods = series_periods.union(period_key)
            period_dict[period_key] = symbol_dict

        # add the data of all symbols for this period (e.g. 2017-Q1)
        series_financials_dict[symbol] = period_dict

save_obj(series_financials_dict, 'dow30_2007-2018_financials')


# ============================================================================================
# Compute the average 3-months price
stock_price_url = Template("https://api.intrinio.com/prices?"
                           "identifier=${symbol}&"
                           "start_date=${start_date}&"
                           "end_date=${end_date}&"
                           "frequency=daily")

report_periods_url = Template("https://api.intrinio.com/fundamentals/standardized?"
                              "identifier=${symbol}&"
                              "statement=income_statement")

series_price_dict = defaultdict(dict)

for symbol in symbols:
    url = report_periods_url.substitute(symbol=symbol)

    data_json = call_and_cache(url)

    for period_info in data_json['data']:
        start_date = period_info['start_date']
        end_date = period_info['end_date']
        year = period_info['fiscal_year']
        fiscal_period = period_info['fiscal_period']
        if fiscal_period in ['Q1', 'Q2', 'Q3', 'Q4']:
            url = stock_price_url.substitute(symbol=symbol, start_date=start_date,
                                             end_date=end_date)
            data_json = call_and_cache(url)

            quarter_mean = np.mean([day_price['close'] for day_price in data_json['data']])

            series_price_dict["%s-%s" % (year, fiscal_period)][symbol] = quarter_mean


save_obj(series_price_dict, 'dow30_2007-2018_T3M_price')



# Create dataset

# Find all possible attributes reported by either of the 3 documents
all_attrs = set()
for symbol in series_financials_dict.keys():
    for period in series_financials_dict[symbol].keys():
        all_attrs = all_attrs.union(list(series_financials_dict[symbol][period].keys()))

# Create a mapping to transform the dicts into matrix the 0 is the stock name
attr2id = {attr: i+1 for i, attr in enumerate(all_attrs)}

