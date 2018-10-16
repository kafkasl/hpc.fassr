import csv
import itertools
import marshal
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from string import Template

import numpy as np
from dateutil.relativedelta import relativedelta

from settings.basic import logging, DATE_FORMAT

from utils import call_and_cache, save_obj, load_obj


class DataCollector(object):
    def __init__(self, symbols_list_name: str, start_year: int, end_year: int):
        # by default we will always gather the stock price of the 3 preceding months, after that we
        # can compute whatever the desired metric is (as of now: current price, 2 weeks, 1 month,
        # 1 quarter)

        self.symbols_list_name = symbols_list_name
        self.start_year = start_year
        self.end_year = end_year
        self.delta_type = 'M'
        self.delta_duration = 3
        self.delta = self._get_relative_delta(type=self.delta_type,
                                              duration=self.delta_duration)
        self.delta_period = "%s%s" % (self.delta_duration, self.delta_type)
        self.data = None

        # Fixed parameters
        self.version = 0.1
        self.basic_symbols = ['symbol',
                              'year',
                              'quarter',
                              'reporting period',
                              'current_date',
                              'next_date',
                              't: curr_half_mean',
                              't: curr_month_mean',
                              't: curr_period_mean',
                              't: current_price',
                              't: daily_increase',
                              't: half_month_increase',
                              't: monthly_increase',
                              't: next_half_mean',
                              't: next_month_mean',
                              't: next_period_mean',
                              't: next_price',
                              't: period_increase']

        attrs = load_obj('../data/intrinio_tags')
        self.extended_symbols = list(set().union(*attrs.values()))

        self.ordered_symbols = self.basic_symbols + self.extended_symbols

        # Common module constants
        self.quarters_names = ["Q1TTM", "Q2TTM", "Q3TTM", "FY"]
        self.statements = ["income_statement", "balance_sheet",
                           "cash_flow_statement", "calculations"]

        self.quarters_idx = [1, 2, 3, 4]
        self.quarter2index = {'Q1TTM': 1, 'Q2TTM': 2, 'Q3TTM': 3, 'FY': 4}

        self.period_keys = ['%s-%s' % (y, q) for y, q in
                            itertools.product(range(2006, 2020),
                                              self.quarters_idx)]

        self.period2index = {period: index for index, period in
                             enumerate(self.period_keys)}
        self.index2period = {index: period for index, period in
                             enumerate(self.period_keys)}

        # Template URLs
        self.fundamentals_url = Template(
            "https://api.intrinio.com/financials/standardized?"
            "identifier=${symbol}&"
            "statement=${statement}&"
            "type=TTM&"
            "fiscal_year=${year}"
            "period=${period}")

        self.stock_price_url = Template("https://api.intrinio.com/prices?"
                                        "identifier=${symbol}&"
                                        "start_date=${start_date}&"
                                        "end_date=${end_date}&"
                                        "frequency=daily")

        self.report_periods_url = Template(
            "https://api.intrinio.com/fundamentals/standardized?"
            "identifier=${symbol}&"
            "statement=income_statement&"
            "type=TTM")

    def csv_filename(self):
        return '../data/csv/%s_monolithic.csv' % self.symbols_list_name

    @staticmethod
    def _get_start_date(end_date_str: str, delta: relativedelta) -> str:
        end_date = datetime.strptime(end_date_str, DATE_FORMAT)

        start_date = end_date - delta

        return start_date.strftime(DATE_FORMAT)

    @staticmethod
    def _load_symbol_list(symbols_list_name: str) -> list:
        return open('../data/%s_symbols.lst' % symbols_list_name).read().split()

    @staticmethod
    def _get_relative_delta(type: str, duration: int) -> relativedelta:
        """
        Get a relative delta
        :param type: M for months, D for days
        :param duration: how long the delta should be
        """
        if type == 'M':
            return relativedelta(months=duration)
        elif type == 'D':
            return relativedelta(days=duration)
        else:
            raise Exception(
                "Unknown type to create timedelta. Expected: [M, D]. Got: [%s]" % type)


            # TODO: remember to update dow30 with ALL symbols ever to be there to avoid survivor bias

    def _collect_attr_names(self, save=True):
        # Parameter dependent variables
        year_range = list(range(self.start_year, self.end_year))
        symbols = DataCollector._load_symbol_list(self.symbols_list_name)

        attr_names = defaultdict(set)

        for symbol in symbols:
            for year in year_range:
                for quarter_name in self.quarters_names:

                    for statement in self.statements:

                        url = self.fundamentals_url.substitute(symbol=symbol,
                                                               statement=statement,
                                                               year=year,
                                                               period=quarter_name)
                        data_json, retries = {}, 3
                        while 'data' not in data_json and retries > 0:

                            data_json = call_and_cache(url)

                            statement_dict = {}

                            for element in data_json['data']:
                                statement_dict[element['tag']] = element[
                                    'value']

                            attr_names[statement].update(statement_dict.keys())

                        if retries == 0:
                            logging.error(
                                "Couldn't get data after 3 retries for url: %s" % url)

        if save:
            save_obj(attr_names,
                     '../data/%s_%s-%s_attr_names' % (self.symbols_list_name,
                                                      self.start_year,
                                                      self.end_year))
        return attr_names

    def _collect_fundamentals(self, save=True):
        # Parameter dependent variables
        year_range = list(range(self.start_year, self.end_year))
        symbols = DataCollector._load_symbol_list(self.symbols_list_name)

        series_financials_dict = {}
        for symbol in symbols:
            period_dict = {}
            for year in year_range:
                for quarter_name in self.quarters_names:

                    symbol_dict = {'symbol': symbol, 'year': year,
                                   'quarter': self.quarter2index[quarter_name]}

                    for statement in self.statements:

                        url = self.fundamentals_url.substitute(symbol=symbol,
                                                               statement=statement,
                                                               year=year,
                                                               period=quarter_name)
                        data_json, retries = {}, 3
                        while 'data' not in data_json and retries > 0:

                            data_json = call_and_cache(url)

                            statement_dict = {}

                            for element in data_json['data']:
                                statement_dict[element['tag']] = element[
                                    'value']

                            symbol_dict.update(statement_dict)

                        if retries == 0:
                            logging.error(
                                "Couldn't get data after 3 retries for url: %s" % url)

                    period_key = "%s-%s" % (
                    year, self.quarter2index[quarter_name])
                    period_dict[self.period2index[period_key]] = symbol_dict

                # add the data of all symbols for this period (e.g. 2017-Q1)
                series_financials_dict[symbol] = period_dict

        if save:
            save_obj(series_financials_dict,
                     '../data/obj/%s_%s-%s_financials' % (
                     self.symbols_list_name,
                     self.start_year,
                     self.end_year))
        return series_financials_dict

    def _get_target_price(self, series_financials_dict: dict,
                          save: bool = True):
        # ============================================================================================
        # Compute some target average (or not) price

        # Parameter dependent variables
        symbols = self._load_symbol_list(self.symbols_list_name)

        series_price_dict = defaultdict(dict)

        for symbol in symbols:
            url = self.report_periods_url.substitute(symbol=symbol)

            data_json = call_and_cache(url)

            for period_info in data_json['data']:

                # TODO: check if this end date is the reporting day or the previous one to avoid
                # future bias (is that the name?)
                reporting_period = "%s - %s" % (
                period_info['start_date'], period_info['end_date'])

                end_date = period_info['end_date']

                start_date = self._get_start_date(end_date, self.delta)

                year = period_info['fiscal_year']
                fiscal_period = period_info['fiscal_period']
                period_key = "%s-%s" % (year, self.quarter2index[fiscal_period])

                # Check that financial data is available (otherwise skip getting stock's price)
                if self.period2index[period_key] in series_financials_dict[
                    symbol]:
                    url = self.stock_price_url.substitute(symbol=symbol,
                                                          start_date=start_date,
                                                          end_date=end_date)
                    data_json = call_and_cache(url)

                    # First entry is oldest one
                    data = list(reversed(data_json['data']))

                    if len(data) > 0:
                        quarter_close_prices = [day_price['close'] for day_price
                                                in data]

                        current_date = data[-1]['date']
                        series_price_dict[symbol][
                            self.period2index[period_key]] = (
                            quarter_close_prices,
                            reporting_period,
                            current_date)

        if save:
            save_obj(series_price_dict,
                     '../data/obj/%s_%s-%s_T%sM_price' % (
                     self.symbols_list_name, self.start_year,
                     self.end_year, self.delta_period))

            return series_price_dict

    def _compute_features(self, series_financials_dict: dict,
                          series_price_dict: dict):
        symbols = self._load_symbol_list(self.symbols_list_name)

        # Write CSVs split by period
        for period_idx in self.period2index.values():

            for symbol in symbols:
                if (period_idx in series_financials_dict[symbol].keys()) and \
                        (period_idx + 1 in series_price_dict[symbol].keys()) and \
                        (period_idx in series_price_dict[symbol].keys()):
                    # series_price is a tuple: ([price1, price2, ...], reporting_period, current_date)
                    curr_period = series_price_dict[symbol][period_idx][0]
                    next_period = series_price_dict[symbol][period_idx + 1][0]

                    rep_period = series_price_dict[symbol][period_idx][1]

                    curr_date = series_price_dict[symbol][period_idx][2]
                    next_date = series_price_dict[symbol][period_idx + 1][2]

                    curr_period_mean = np.mean(curr_period)
                    next_period_mean = np.mean(next_period)

                    curr_month_mean = np.mean(curr_period[-30:])
                    next_month_mean = np.mean(next_period[-30:])

                    curr_half_mean = np.mean(curr_period[-15:])
                    next_half_mean = np.mean(next_period[-15:])

                    curr_price = curr_period[-1]
                    next_reporting_price = next_period[-1]

                    period_incr = (next_period_mean / curr_period_mean) - 1
                    monthly_incr = (next_month_mean / curr_month_mean) - 1

                    half_incr = (next_half_mean / curr_half_mean) - 1
                    daily_incr = (next_reporting_price / curr_price) - 1

                    extra_fields = {'next_date': next_date,
                                    'current_date': curr_date,
                                    'reporting period': rep_period,
                                    't: period_increase': period_incr,
                                    't: monthly_increase': monthly_incr,
                                    't: half_month_increase': half_incr,
                                    't: daily_increase': daily_incr,
                                    't: current_price': curr_price,
                                    't: next_price': next_reporting_price,
                                    't: curr_half_mean': curr_half_mean,
                                    't: next_half_mean': next_half_mean,
                                    't: curr_month_mean': curr_month_mean,
                                    't: next_month_mean': next_month_mean,
                                    't: curr_period_mean': curr_period_mean,
                                    't: next_period_mean': next_period_mean,
                                    }

                    series_financials_dict[symbol][period_idx].update(
                        extra_fields)
                elif period_idx in series_financials_dict[symbol].keys():
                    del series_financials_dict[symbol][period_idx]

        return series_financials_dict

    @staticmethod
    def _get_attr2id(series_financials_dict: dict) -> dict:
        all_attrs = set()
        for symbol in series_financials_dict.keys():
            for period in series_financials_dict[symbol].keys():
                all_attrs = all_attrs.union(
                    list(series_financials_dict[symbol][period].keys()))

        attr2id = {attr: i + 1 for i, attr in enumerate(all_attrs)}

        return attr2id

    def _write_to_single_csv(self, series_financials_dict: dict):
        file_name = self.csv_filename()
        attr2id = self._get_attr2id(
            series_financials_dict=series_financials_dict)
        symbols = self._load_symbol_list(self.symbols_list_name)

        assert len(sorted(self.ordered_symbols)) >= len(sorted(attr2id.keys()))

        with open(file_name, mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.ordered_symbols)

            writer.writeheader()

            for period_idx in self.period2index.values():

                for symbol in symbols:

                    if period_idx in series_financials_dict[symbol].keys():
                        row = series_financials_dict[symbol][period_idx]
                        writer.writerow(row)

        return file_name

    def collect(self) -> dict:

        if not self.data:

            series_financials = self._collect_fundamentals()

            series_prices = self._get_target_price(
                series_financials_dict=series_financials)

            self.data = self._compute_features(
                series_financials_dict=series_financials,
                series_price_dict=series_prices)
        else:
            print("Data already collected.")
        return deepcopy(self.data)

    def to_csv(self) -> str:
        file_csv = self._write_to_single_csv(series_financials_dict=self.data)
        return file_csv

    def get_log(self):
        attrs = self.__dict__.keys()

        attrs.update({'class_code': marshal.dumps(DataCollector)})

        return attrs


if __name__ == '__main__':
    symbols_list_name = 'dow30'
    # symbols_list_name = 'debug'
    start_year = 2006
    end_year = 2019

    dc = DataCollector(symbols_list_name=symbols_list_name,
                       start_year=start_year,
                       end_year=end_year)

    attrs = dc._collect_attr_names()
    # data = dc.collect()
    # filename = dc.to_csv()

# data = final_dict
# data = final_dict['TSLA']
# ["[%s] -> [%s] %s - %s" % (
#     e['reporting period'], e['next_date'], e['t: current_price'], e['t: next_price']) for e in
#  data.values()]
