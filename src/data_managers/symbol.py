from utils.cache import call_and_cache
from string import Template
from datetime import datetime

from settings.basic import *

year_tag_url = Template("https://api.intrinio.com/historical_data?"
                        "identifier=${symbol}&"
                        "item=${tag}&"
                        "frequency=${type}&"
                        "start_date=${start}&"
                        "end_date=${end}")

stock_price_url = Template("https://api.intrinio.com/prices?"
                           "identifier=${symbol}&"
                           "start_date=${start_date}&"
                           "end_date=${end_date}&"
                           "frequency=daily")


class Symbol(object):
    indicators = ('totalrevenue', 'totalliabilities', 'totalassets', 'basiceps',
                  'dividendyield', 'bookvaluepershare', 'nwc', 'pricetoearnings', 'pricetobook')

    def __init__(self, symbol: str):
        self.id = symbol
        self.name = ""
        self._data = {}


    def _get_yearly_tag_values(self, tag):
        start_date = datetime(year=2007, month=1, day=1).strftime(DATE_FORMAT)
        end_date = datetime.today().strftime(DATE_FORMAT)

        url = year_tag_url.substitute(symbol=self.id, tag=tag, type='yearly', start=start_date,
                                      end=end_date)

        return call_and_cache(url)

    def get_indicator(self, indicator: str):
        assert indicator in self.indicators

        if indicator not in self._data:
            data_json = self._get_yearly_tag_values(indicator)

            self._data[indicator] = {}
            for r in data_json['data']:
                try:
                    self._data[indicator][int(r['date'][0:4])] = float(r['value'])
                except ValueError:
                    logging.error("[%s] Indicator %s value is not a number [%s]" %
                                  (self.id, indicator, data_json['data']))

        return self._data[indicator]

    def get_y(self, indicator: str, year: int):
        """
        :return: dictionary mapping year -> total revenue
        """

        values = self.get_indicator(indicator=indicator)

        try:
            return values[year]
        except KeyError as e:
            raise KeyError("Accessing indicator %s of symbol %s with year %s, but dict "
                           "is: %s %s" % (indicator, self.id, year, values, e))

    def get_stock_price(self, date: datetime):

        date_str = date.strftime(DATE_FORMAT)

        if 'open_price' not in self._data:
            self._data['open_price'] = {}
        if date_str not in self._data['open_price']:
            url = stock_price_url.substitute(symbol=self.id, start_date=date_str,
                                             end_date=date_str)

            data_json = call_and_cache(url)

            assert len(data_json['data']) == 1

            self._data['open_price'][data_json['data'][0]['date']] = data_json['data'][0]['open']

        return self._data['open_price'][date_str]
