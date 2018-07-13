import requests
import json
import os
import base64

import pandas as pd

from requests.auth import HTTPBasicAuth
from settings.basic import (INTRINIO_CACHE_PATH, intrinio_username, intrinio_password, DATE_FORMAT)
from string import Template
from datetime import datetime
from functools import reduce

_fundamental_template = Template('https://api.intrinio.com/historical_data?identifier=${'
                                 'symbol}&item='
                                 '${tag}&start_date=${start_date}&end_date=${end_date}')


# _revenue_template = Template('https://api.intrinio.com/historical_data?identifier=${symbol}&item='
#                              '${tag}&start_date=${start_date}&end_date=${end_date}&frequency=${'
#                              'frequency}&type=${type}')


def get_tag(symbol: str, tag: str, start_date: datetime, end_date: datetime, frequency: str = '',
            type: str = ''):
    """

    :param symbol:
    :param start_date: (datetime) the earliest date for which to return data: YYYY-MM-DD
    :param end_date: (datetime) the latest date for which to return data: YYYY-MM-DD
    :param frequency: the frequency of the historical prices & valuation data:
     [daily | weekly | monthly | quarterly | yearly]
    :param type: (optional, returns trailing twelve months (TTM) for the income statement,
     cash flow statement and calculations, and quarterly (QTR) for balance sheet) - the type of
     periods requested - includes fiscal years for annual data, quarters for quarterly data and
     trailing twelve months for annual data on a quarterly basis: FY | QTR | TTM | YTD
    """
    # format days according to YYYY-MM-DD
    start_date = start_date.strftime(DATE_FORMAT)
    end_date = end_date.strftime(DATE_FORMAT)

    url = _fundamental_template.substitute(symbol=symbol, tag=tag, start_date=start_date,
                                           end_date=end_date)

    cached_file = os.path.join(INTRINIO_CACHE_PATH, base64.standard_b64encode(url.encode(

    )).decode())

    print("Cached file: %s" % cached_file)
    if os.path.exists(cached_file):
        print("Data was present in cache, loading: %s" % cached_file)
        data_json = json.loads(open(cached_file, 'r').read())
    else:
        print("Data was not present in cache calling request: %s" % url)
        r = requests.get(url, auth=HTTPBasicAuth(intrinio_username, intrinio_password))

        if r.status_code != 200:
            raise Exception("ERROR: Request status was: %s\nRequest URL: %s" % (r.status_code, url))
        data_json = json.loads(r.text)

        with open(cached_file, 'w') as f:
            f.write(json.dumps(data_json))
            print("Successfully cached url: %s to %s" % (url, cached_file))
    return data_json


def to_df(symbol, data_json):
    df = pd.DataFrame(data_json['data'])

    df['date'] = pd.to_datetime(df['date'], format=DATE_FORMAT)
    df['symbol'] = symbol

    df = df.rename(columns={'value': data_json['item']})
    # df.set_index(pd.DatetimeIndex(df['date']), inplace=True)

    return df


def build_df_for_graham(symbol: str, start_date: datetime, end_date: datetime, frequency: str = '',
                        type: str = ''):
    tags = ['totalrevenue', 'totalassets', 'totalliabilities', 'basiceps', 'paymentofdividends',
            'pricetoearnings', 'bookvaluepershare']

    ds = []
    for tag in tags:
        data_json = get_tag(symbol=symbol, tag=tag, start_date=start_date, end_date=end_date,
                       frequency=frequency, type=type)
        ds.append(to_df(symbol, data_json))

    df = reduce(
        lambda left, right: pd.merge(left, right, on=['symbol', 'date'], how='outer'), ds)
    df.set_index(['date', 'symbol'], inplace=True)

    return df

if __name__ == '__main__':
    url = 'https://api.intrinio.com/companies?ticker=AAPL'


    # r = requests.get(url, auth=HTTPBasicAuth(username, password))

    # print("R: %s" % r)

    # download fundamental from:
    # https: // intrinio.com / data / us - fundamentals - financials - metrics - ratios
