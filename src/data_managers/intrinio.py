from settings.basic import (logging, DATE_FORMAT)
from string import Template
from datetime import datetime
from functools import reduce
from data_managers.symbol import Symbol
from utils.cache import call_and_cache

import pandas as pd

_symbols_url = Template("https://api.intrinio.com/companies?page_number=${page_number}")
_fundamental_template = Template(
    'https://api.intrinio.com/historical_data?identifier=${symbol}&item=${tag}'
    '&start_date=${start_date}&end_date=${end_date}&frequency=${frequency}')
base_url = Template('https://api.intrinio.com/financials/standardized?identifier=${symbol}'
                    '&statement=${doc}&fiscal_year=${year}&fiscal_period=${period}')


def build_symbols(tickers: list) -> list:
    return {t: Symbol(t) for t in tickers}








# PROBABLY DEPRECATED




def get_all_symbols(**kwargs) -> list:
    curr_url = _symbols_url.substitute(page_number=1)
    data_json = call_and_cache(curr_url, **kwargs)

    symbols = [r['ticker'] for r in data_json['data']]

    total_pages = int(data_json['total_pages'])

    for page_number in range(2, total_pages + 1):
        curr_url = _symbols_url.substitute(page_number=page_number)
        data_json = call_and_cache(curr_url, **kwargs)

        symbols.extend([r['ticker'] for r in data_json['data']])

    return symbols


# TODO: check if type affects
def get_tag(symbol: str, tag: str, start_date: datetime, end_date: datetime,
            frequency: str, **kwargs) -> dict:
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
                                           end_date=end_date, frequency=frequency)

    data_json = call_and_cache(url, **kwargs)

    return data_json


def to_df(symbol: str, data_json: dict) -> pd.DataFrame:
    df = pd.DataFrame(data_json['data'])

    df['date'] = pd.to_datetime(df['date'], format=DATE_FORMAT)
    df['symbol'] = symbol

    df = df.rename(columns={'value': data_json['item']})
    # df.set_index(pd.DatetimeIndex(df['date']), inplace=True)

    return df


def build_df_for_tags(symbol: str, tags: list, start_date: datetime, end_date: datetime,
                      frequency: str = 'quarterly', type_opt: str = '', **kwargs) -> pd.DataFrame:
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
    start_date = datetime(year=2007, month=1, day=1)
    end_date = datetime.now()
    tags = ['totalrevenue', 'totalassets', 'totalliabilities', 'basiceps', 'paymentofdividends',
            'pricetoearnings', 'bookvaluepershare']
    df_list = []
    for symbol in symbols:
        try:
            df_aux = build_df_for_tags(symbol=symbol, tags=tags, start_date=start_date,
                                       end_date=end_date)
            if df_aux.empty:
                logging.info("Symbol %s returned no fundamental data." % symbol)
            else:
                logging.info("Successfully loaded fundamental data for symbol %s." % symbol)
                df_list.append(df_aux)
        except KeyError as e:
            logging.error("Could not get fundamental data for %s, exception: %s" % (symbol, e))

    df = pd.concat(df_list)

    return df


def build_year(symbol: str, year: str) -> pd.DataFrame:
    quarters = ["Q1TTM", "Q2TTM", "Q3TTM", "FY"]

    documents = ["income_statement", "balance_sheet", "cash_flow_statement", "calculations"]

    df_list = []
    for quarter in quarters:
        docs_aux = [pd.DataFrame([['symbol', symbol], ['year', year], ['quarter',
                                                                       quarters.index(
                                                                           quarter) + 1]],
                                 columns=['tag', 'value']).set_index('tag')]
        for doc in documents:
            curr_url = base_url.substitute(symbol=symbol, doc=doc, year=year,
                                           period=quarter)
            data_json = call_and_cache(curr_url)
            df_aux = pd.DataFrame(data_json['data']).drop_duplicates()
            df_aux['tag'] = ''.join([x[0] for x in doc.split('_')]) + '_' + df_aux[
                'tag'].astype(str)
            docs_aux.append(df_aux.set_index('tag'))

        df_list.append(pd.concat(docs_aux, axis=0).drop_duplicates().T)

    return df_list
    # return pd.concat(df_list, axis=0, sort=True).set_index(['symbol', 'year', 'quarter'])


def get_all_fundamental_data(symbols, years):
    tags = ['totalrevenue', 'totalassets', 'totalliabilities', 'basiceps', 'paymentofdividends',
            'pricetoearnings', 'bookvaluepershare']

    symbols_df = {}
    for symbol in symbols:
        df_list = []
        for year in years:
            df_list.extend(build_year(symbol, year))
        df = pd.concat(df_list, axis=0).set_index(['symbol', 'year', 'quarter'])

        symbols_df[symbol] = df

    # pd.isnull(df).sum() > 0 check if a columns has null values
    return symbols_df


if __name__ == '__main__':
    url = 'https://api.intrinio.com/companies?ticker=AAPL'


    # r = requests.get(url, auth=HTTPBasicAuth(username, password))

    # logging.info("R: %s" % r)

    # download fundamental from:
    # https: // intrinio.com / data / us - fundamentals - financials - metrics - ratios
