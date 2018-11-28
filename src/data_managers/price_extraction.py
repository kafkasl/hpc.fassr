from string import Template

import pandas as pd

from utils import call_and_cache, load_symbol_list


class PriceExtractor(object):
    stock_price_url = Template("https://api.intrinio.com/prices?"
                               "identifier=${symbol}&"
                               "start_date=${start_date}&"
                               "page_number=${page_number}&"
                               "frequency=daily&sort_order=asc")

    def __init__(self, symbols_list_name: str = 'dow30',
                 start_date='2006-01-01'):

        self.symbols_list_name = symbols_list_name
        self.start_date = start_date

    @staticmethod
    def _prices_to_list(data: list):
        prices = []
        for day in data:
            prices.append((day['date'], day['adj_close']))

        return prices

    @staticmethod
    def _prices_to_daframe(symbol: str, prices: list) -> pd.DataFrame:
        df = pd.DataFrame(prices, columns=['date', 'price'])
        # df['date'] = pd.to_datetime(df['date'], format=DATE_FORMAT)
        df['symbol'] = symbol
        # df = df.set_index(['date', 'symbol'])

        return df

    def _get_symbol_prices(self, symbol: str) -> pd.DataFrame:
        url = self.stock_price_url.substitute(symbol=symbol,
                                              start_date=self.start_date,
                                              page_number=1)
        data_json = call_and_cache(url)

        total_pages = data_json['total_pages']
        page_number = 2
        prices = self._prices_to_list(data_json['data'])

        while page_number <= total_pages:
            url = self.stock_price_url.substitute(symbol=symbol,
                                                  start_date=self.start_date,
                                                  page_number=page_number)
            data_json = call_and_cache(url)

            prices.extend(self._prices_to_list(data_json['data']))

            page_number += 1

        return self._prices_to_daframe(symbol=symbol, prices=prices)

    def collect(self) -> pd.DataFrame:
        symbols = load_symbol_list(self.symbols_list_name)

        dfs = []

        for symbol in symbols:
            dfs.append(
                self._get_symbol_prices(symbol=symbol))

        return pd.concat(dfs)


if __name__ == "__main__":
    symbols_list_name = 'sp500'
    start_date = '2006-01-01'

    df = PriceExtractor(symbols_list_name=symbols_list_name,
                        start_date=start_date).collect()
