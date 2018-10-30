import csv
import itertools
import marshal
import os
from collections import defaultdict
from datetime import datetime
from string import Template

import pandas as pd
from dateutil.relativedelta import relativedelta

from settings.basic import logging, DATE_FORMAT, DATA_PATH, CACHE_ENABLED
from tags import Tags
from utils import call_and_cache, save_obj, load_symbol_list


class FundamentalsCollector(object):
    def __init__(self, symbols_list_name: str, start_year: int, end_year: int,
                 cache: bool = CACHE_ENABLED):

        self.symbols_list_name = symbols_list_name
        self.start_year = start_year
        self.end_year = end_year
        self.cache = cache

        self.version = 0.2

        # attrs = load_obj('../data/intrinio_tags_%s' % symbols_list_name)
        # self.extended_symbols = list(set().union(*attrs.values()))

        # self.ordered_symbols = self.basic_symbols + self.extended_symbols
        self.ordered_symbols = Tags.all()

        # Common module constants
        self.quarters_names = ["Q1", "Q2", "Q3", "Q4"]
        self.statements = ["income_statement", "balance_sheet",
                           "cash_flow_statement", "calculations"]
        self.year_range = list(range(self.start_year, self.end_year))

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
            "type=QTR")

    @property
    def csv_filename(self):
        return '%s/csv/%s_monolithic.csv' % (DATA_PATH, self.symbols_list_name)

    @staticmethod
    def _get_start_date(end_date_str: str, delta: relativedelta) -> str:
        end_date = datetime.strptime(end_date_str, DATE_FORMAT)

        start_date = end_date - delta

        return start_date.strftime(DATE_FORMAT)

        # TODO: remember to update dow30 with ALL symbols ever to be
        #  there to avoid survivor bias

    def _collect_attr_names(self, save=True):
        # Parameter dependent variables

        symbols = load_symbol_list(self.symbols_list_name)

        attr_names = defaultdict(set)

        for symbol in symbols:
            for year in self.year_range:
                for quarter_name in self.quarters_names:

                    for statement in self.statements:

                        url = self.fundamentals_url.substitute(symbol=symbol,
                                                               statement=statement,
                                                               year=year,
                                                               period=quarter_name)
                        data_json, retries = {}, 3
                        while 'data' not in data_json and retries > 0:

                            data_json = call_and_cache(url, cache=self.cache)

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
                     '%s/%s_%s-%s_attr_names' % (DATA_PATH,
                                                 self.symbols_list_name,
                                                 self.start_year,
                                                 self.end_year))
        return attr_names

    def _collect_fundamentals(self, save=True):
        # Parameter dependent variables
        year_range = list(range(self.start_year, self.end_year))
        symbols = load_symbol_list(self.symbols_list_name)

        series_financials_dict = {}
        for symbol in symbols:
            period_dict = {}
            for year in year_range:
                for quarter in self.quarters_names:

                    symbol_dict = {Tags.symbol: symbol,
                                   Tags.year: year,
                                   Tags.quarter: quarter}

                    for statement in self.statements:

                        url = self.fundamentals_url \
                            .substitute(symbol=symbol,
                                        statement=statement,
                                        year=year,
                                        period=quarter)

                        data_json, retries = {}, 3
                        while 'data' not in data_json and retries > 0:

                            data_json = call_and_cache(url, cache=self.cache)

                            statement_dict = {}

                            if 'data' in data_json:
                                for element in data_json['data']:
                                    statement_dict[element['tag']] = element[
                                        'value']

                                symbol_dict.update(statement_dict)

                        if retries == 0:
                            logging.error(
                                "Couldn't get data after 3 retries for url: %s" % url)

                    period_dict['{}{}'.format(year, quarter)] = symbol_dict

                series_financials_dict[symbol] = period_dict

        if save:
            save_obj(series_financials_dict,
                     '%s/obj/%s_%s-%s_financials' % (
                         DATA_PATH,
                         self.symbols_list_name,
                         self.start_year,
                         self.end_year))
        return series_financials_dict

    def _add_periods_info(self, series_financials_dict: dict,
                          save: bool = True):
        # ============================================================================================
        # Compute some target average (or not) price

        # Parameter dependent variables
        symbols = load_symbol_list(self.symbols_list_name)

        for symbol in symbols:
            url = self.report_periods_url.substitute(symbol=symbol)

            data_json = call_and_cache(url, cache=self.cache)

            reporting_periods = data_json['data']
            for i in range(len(reporting_periods)):

                period_info = reporting_periods[i]

                start_date = period_info['start_date']
                end_date = period_info['end_date']
                quarter = period_info['fiscal_period']
                year = period_info['fiscal_year']

                # Starting date of the next reporting period
                next_date = (datetime.strptime(end_date, DATE_FORMAT) +
                             relativedelta(months=3)).strftime(DATE_FORMAT)
                if i > 0:
                    next_date = reporting_periods[i - 1]['start_date']

                # Starting date of the previous reporting period
                prev_date = (datetime.strptime(start_date, DATE_FORMAT) -
                             relativedelta(months=3)).strftime(DATE_FORMAT)
                if i < len(reporting_periods) - 1:
                    prev_date = reporting_periods[i + 1]['start_date']

                period_dict = {Tags.symbol: symbol,
                               Tags.quarter: quarter,
                               Tags.year: year,
                               Tags.current_date: end_date,
                               Tags.prev_date: prev_date,
                               Tags.next_date: next_date,
                               Tags.rep_period: '{}:{}'
                                   .format(start_date, end_date),
                               Tags.next_rep_period: '{}:{}'
                                   .format(end_date, next_date)}

                period_key = '{}{}'.format(year, quarter)
                try:
                    series_financials_dict[symbol][period_key].update(period_dict)
                except KeyError:
                    logging.warning("No fundamental info for symbol %s in period %s" % (symbol, period_key))

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
        attr2id = self._get_attr2id(
            series_financials_dict=series_financials_dict)
        symbols = load_symbol_list(self.symbols_list_name)

        assert len(sorted(self.ordered_symbols)) >= len(sorted(attr2id.keys()))

        with open(self.csv_filename, mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.ordered_symbols)

            writer.writeheader()

            period_indices = ['{}{}'.format(year, quarter) for year, quarter in
                              itertools.product(self.year_range,
                                                self.quarters_names)]
            for idx in period_indices:

                for symbol in symbols:

                    symbol_dict = series_financials_dict[symbol]
                    if idx in symbol_dict.keys() and \
                                    Tags.current_date in symbol_dict[idx]:
                        row = symbol_dict[idx]
                        writer.writerow(row)

        return self.csv_filename

    def collect(self) -> pd.DataFrame:

        if self.cache and os.path.isfile(self.csv_filename):
            dataframe = pd.read_csv(self.csv_filename)

        else:
            series_financials = self._collect_fundamentals()

            self._add_periods_info(series_financials_dict=series_financials)

            self._to_csv(series_financials)

            dataframe = pd.read_csv(self.csv_filename)

        return dataframe

    def _to_csv(self, data) -> str:
        file_csv = self._write_to_single_csv(series_financials_dict=data)
        return file_csv

    def get_log(self):
        attrs = self.__dict__.keys()

        attrs.update({'class_code': marshal.dumps(FundamentalsCollector)})

        return attrs


if __name__ == '__main__':
    symbols_list_name = 'sp500'
    # symbols_list_name = 'debug'
    start_year = 2006
    end_year = 2019

    df_fund = FundamentalsCollector(symbols_list_name=symbols_list_name,
                                    start_year=start_year,
                                    end_year=end_year, cache=False).collect()
