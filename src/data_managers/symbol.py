from utils.cache import call_and_cache
from string import Template
from datetime import date

from settings.basic import *

base_url = Template("https://api.intrinio.com/historical_data?"
                    "identifier=${symbol}&"
                    "item=${tag}&"
                    "type=${type}&"
                    "start_date=${start}&"
                    "end_date=${end}")

class Symbol(object):
    def __init__(self, symbol: str):
        self.id = symbol
        self._total_revenue = None
        self._total_liabilities = None
        self._total_assets = None
        self._total_eps = None
        self._dividend_yield = None
        self._book_value_per_share = None

    def _get_yearly_tag(self, tag):
        start_date = date(year=2007, month=1, day=1).strftime(DATE_FORMAT)
        end_date = date.today().strftime(DATE_FORMAT)

        url = base_url.substitute(symbol=self.id, tag=tag, start=start_date, end=end_date)

        return call_and_cache(url)



    @property
    def total_revenue(self):
        """
        :return: dictionary mapping year -> total revenue
        """
        if not self._total_revenue:
            tag = 'totalrevenue'

            data_json = self._get_yearly_tag(tag)

            self._total_revenue = {int(row['date'][0:4]): row['value'] for row in data_json[
                'data']}

        return self._total_revenue

    @property
    def total_liabilities(self):
        """
        :return: dictionary mapping year -> total liabilities
        """
        if not self._total_liabilities:
            tag = 'totalliabilities'

            data_json = self._get_yearly_tag(tag)

            self._total_liabilities = {int(row['date'][0:4]): row['value'] for row in data_json[
                'data']}

        return self._total_liabilities

    @property
    def total_assets(self):
        """
        :return: dictionary mapping year -> total assets
        """
        if not self._total_assets:
            tag = 'totalassets'

            data_json = self._get_yearly_tag(tag)

            self._total_assets = {int(row['date'][0:4]): row['value'] for row in data_json[
                'data']}

        return self._total_assets

    @property
    def basiceps(self):
        """
        :return: dictionary mapping year -> basic eps
        """
        if not self._total_eps:
            tag = 'basiceps'

            data_json = self._get_yearly_tag(tag)

            self._total_eps = {int(row['date'][0:4]): row['value'] for row in data_json[
                'data']}

        return self._total_eps

    @property
    def dividend_yield(self):
        """
        :return: dictionary mapping year -> dividend yield
        """
        if not self._dividend_yield:
            tag = 'dividendyield'

            data_json = self._get_yearly_tag(tag)

            self._dividend_yield = {int(row['date'][0:4]): row['value'] for row in data_json[
                'data']}

        return self._dividend_yield

    @property
    def book_value_per_share(self):
        """
        :return: dictionary mapping year -> book value per share
        """
        if not self._book_value_per_share:
            tag = 'bookvaluepershare'

            data_json = self._get_yearly_tag(tag)

            self._book_value_per_share = {int(row['date'][0:4]): row['value'] for row in data_json[
                'data']}

        return self._book_value_per_share