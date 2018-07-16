import requests
import json
import os
import base64

import pandas as pd

from requests.auth import HTTPBasicAuth
from settings.basic import (iio_symbols, logging, CACHE_PATH, intrinio_username,
                            intrinio_password,
                            DATE_FORMAT)
from string import Template
from datetime import datetime
from functools import reduce
from typing import Tuple
from urllib.parse import urlparse

_symbols_url = Template("https://api.intrinio.com/companies?page_number=${page_number}")
_fundamental_template = Template(
    'https://api.intrinio.com/historical_data?identifier=${symbol}&item=${tag}'
    '&start_date=${start_date}&end_date=${end_date}&frequency=${frequency}')


# _revenue_template = Template('https://api.intrinio.com/historical_data?identifier=${symbol}&item='
#                              '${tag}&start_date=${start_date}&end_date=${end_date}&frequency=${'
#                              'frequency}&type=${type}')


def _call_and_cache(url: str, **kwargs) -> dict:
    url_parsed = urlparse(url)

    cached_file = os.path.join(CACHE_PATH, url_parsed.netloc + url_parsed.path + "/" +
                               base64.standard_b64encode(url_parsed.query.encode()).decode())

    if not os.path.exists(os.path.dirname(cached_file)):
        os.makedirs(os.path.dirname(cached_file))

    try:
        no_cache = kwargs['no-cache']
    except KeyError:
        no_cache = False

    data_json = {}
    if os.path.exists(cached_file) and not no_cache:
        logging.debug("Data was present in cache and cache is enabled, loading: %s" % cached_file)
        with open(cached_file, 'r') as f:
            data_json = json.loads(f.read())
    else:
        logging.info(
            "Data was either not present in cache or it was disabled calling request: %s" % url)
        r = requests.get(url, auth=HTTPBasicAuth(intrinio_username, intrinio_password))

        if r.status_code != 200:
            logging.error("Request status was: %s for URL: %s" % (r.status_code, url))
            return data_json

        data_json = json.loads(r.text)

        if not len(data_json['data']) > 0:
            logging.debug("Data field is empty.\nRequest URL: %s" % (url))

        with open(cached_file, 'w') as f:
            f.write(json.dumps(data_json))
            logging.debug("Successfully cached url: %s to %s" % (url, cached_file))

    return data_json


def get_symbols(**kwargs) -> list:
    curr_url = _symbols_url.substitute(page_number=1)
    data_json = _call_and_cache(curr_url, **kwargs)

    symbols = [r['ticker'] for r in data_json['data']]

    total_pages = int(data_json['total_pages'])

    for page_number in range(2, total_pages + 1):
        curr_url = _symbols_url.substitute(page_number=page_number)
        data_json = _call_and_cache(curr_url, **kwargs)

        symbols.extend([r['ticker'] for r in data_json['data']])

    return symbols


# TODO: check if type affects
def get_tag(symbol: str, tag: str, start_date: datetime, end_date: datetime,
            frequency: str, type_opt: str, **kwargs) -> dict:
    """

    :param type_opt:
    :param symbol:
    :param tag: tag of the data point to retrieve
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

    print("Frequency: %s" % frequency)
    url = _fundamental_template.substitute(symbol=symbol, tag=tag, start_date=start_date,
                                           end_date=end_date, frequency=frequency, type= type_opt)

    data_json = _call_and_cache(url, **kwargs)

    return data_json


def to_df(symbol: str, data_json: dict) -> pd.DataFrame:
    df = pd.DataFrame(data_json['data'])

    df['date'] = pd.to_datetime(df['date'], format=DATE_FORMAT)
    df['symbol'] = symbol

    df = df.rename(columns={'value': data_json['item']})
    # df.set_index(pd.DatetimeIndex(df['date']), inplace=True)

    return df


def build_df_for_graham(symbol: str, start_date: datetime, end_date: datetime, frequency: str =
'quarterly', type_opt: str = '', **kwargs) -> pd.DataFrame:
    tags = ['totalrevenue', 'totalassets', 'totalliabilities', 'basiceps', 'paymentofdividends',
            'pricetoearnings', 'bookvaluepershare']

    ds = []
    for tag in tags:
        data_json = get_tag(symbol=symbol, tag=tag, start_date=start_date, end_date=end_date,
                            frequency=frequency, type_opt=type_opt, **kwargs)
        df_aux = to_df(symbol, data_json)
        if df_aux.empty:
            logging.warning(
                "Symbol %s had not fundamental data for tag %s. Skipping." % (symbol, tag))
            return pd.DataFrame()
        ds.append(df_aux)

    df = reduce(
        lambda left, right: pd.merge(left, right, on=['symbol', 'date'], how='outer'), ds)
    df.set_index(['date', 'symbol'], inplace=True)

    return df


def merge_quarter(quarter_lst):
    df = pd.concat(quarter_lst, axis=0).drop_duplicates().T

    return df


def get_all_fundamental_data_deprecated(symbols):
    start_date = datetime(year=1900, month=3, day=31)
    end_date = datetime.now()

    df_list = []
    for symbol in symbols:
        try:
            df_aux = build_df_for_graham(symbol=symbol, start_date=start_date, end_date=end_date)
            if df_aux.empty:
                logging.info("Symbol %s returned no fundamental data." % symbol)
            else:
                logging.info("Successfully loaded fundamental data for symbol %s." % symbol)
                df_list.append(df_aux)
        except KeyError as e:
            logging.error("Could not get fundamental data for %s, exception: %s" % (symbol, e))

    df = pd.concat(df_list)

    return df


def get_all_fundamental_data(symbols):
    # start_date = datetime(year=1900, month=3, day=31)
    # end_date = datetime.now()
    years = [2014, 2015, 2016, 2017]
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    documents = ["income_statement", "balance_sheet", "cash_flow_statement"]
    # documents = ["income_statement", "balance_sheet", "cash_flow_statement", "calculations"]
    tags = ['totalrevenue', 'totalassets', 'totalliabilities', 'basiceps', 'paymentofdividends',
            'pricetoearnings', 'bookvaluepershare']

    base_url = Template(
        'https://api.intrinio.com/financials/standardized?identifier=${symbol}&statement=${doc}&fiscal_year=${year}&fiscal_period=${period}')
    df_list = []

    for symbol in symbols:
        year_aux = {}
        for year in years:
            quarter_aux = {}
            for quarter in quarters:
                docs_aux = [pd.DataFrame([['symbol', symbol], ['year', year], ['quarter',
                                                                               quarter]],
                                         columns=['tag', 'value']).set_index('tag')]
                for doc in documents:
                    curr_url = base_url.substitute(symbol=symbol, doc=doc, year=year,
                                                   period=quarter)
                    data_json = _call_and_cache(curr_url)
                    df_aux = pd.DataFrame(data_json['data']).drop_duplicates()
                    df_aux['tag'] = ''.join([x[0] for x in doc.split('_')]) + '_' + df_aux[
                        'tag'].astype(str)
                    docs_aux.append(df_aux.set_index('tag'))

                quarter_aux[quarter] = docs_aux
                quarter_df = pd.concat(docs_aux, axis=0).drop_duplicates().T
                df_list.append(quarter_df)
            year_aux[year] = quarter_aux

    df = pd.concat(df_list, axis=0).set_index(['symbol', 'year', 'quarter'])

    # pd.isnull(df).sum() > 0 check if a columns has null values
    return df


if __name__ == '__main__':
    url = 'https://api.intrinio.com/companies?ticker=AAPL'


    # r = requests.get(url, auth=HTTPBasicAuth(username, password))

    # logging.info("R: %s" % r)

    # download fundamental from:
    # https: // intrinio.com / data / us - fundamentals - financials - metrics - ratios
