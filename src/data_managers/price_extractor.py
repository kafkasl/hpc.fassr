from string import Template

import pandas as pd

from settings.basic import DATE_FORMAT
from utils import call_and_cache

stock_price_url = Template("https://api.intrinio.com/prices?"
                           "identifier=${symbol}&"
                           "start_date=${start_date}&"
                           "page_number=${page_number}&"
                           "frequency=daily&sort_order=asc")


def prices_to_list(data: list):
    prices = []
    for day in data:
        prices.append((day['date'], day['close']))

    return prices


def prices_to_daframe(symbol: str, prices: list) -> pd.DataFrame:
    df = pd.DataFrame(prices, columns=['date', 'price'])
    # df['date'] = pd.to_datetime(df['date'], format=DATE_FORMAT)
    df['symbol'] = symbol
    # df = df.set_index(['date', 'symbol'])

    return df


def get_symbol_prices(symbol: str, start_date: str) -> pd.DataFrame:
    url = stock_price_url.substitute(symbol=symbol, start_date=start_date,
                                     page_number=1)
    data_json = call_and_cache(url)

    total_pages = data_json['total_pages']
    page_number = 2
    prices = prices_to_list(data_json['data'])

    while page_number <= total_pages:
        url = stock_price_url.substitute(symbol=symbol, start_date=start_date,
                                         page_number=page_number)
        data_json = call_and_cache(url)

        prices.extend(prices_to_list(data_json['data']))

        page_number += 1

    return prices_to_daframe(symbol=symbol, prices=prices)


def get_prices(symbols: list, start_date='2006-01-01') -> pd.DataFrame:

    dfs = []

    for symbol in symbols:
        dfs.append(get_symbol_prices(symbol=symbol, start_date=start_date))

    return pd.concat(dfs)


if __name__ == "__main__":
    symbols_list_name = 'dow30'
    start_date = '2006-01-01'
    symbols = open('../data/%s_symbols.lst' % symbols_list_name).read().split()

    df = get_prices(symbols=symbols, start_date=start_date)
