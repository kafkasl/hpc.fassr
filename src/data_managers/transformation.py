# basically converts the fundamental variables into actual indicators
import marshal
from math import sqrt

import numpy as np


# 'totalrevenue', 'totalliabilities', 'totalassets', 'basiceps',
# 'dividendyield', 'bookvaluepershare', 'nwc', 'pricetoearnings', 'pricetobook'

# totalcurrentliabilities: The total current liabilities is the sum of all liabilities of the company due in less than 1 year
# totalliabilities: Total liabilities are the total amounts due to all parties to the company including vendors, employees, bondholders, lenders and more.
# totalcurrentassets: Current assets are balance sheet accounts that represent the value of all assets that can reasonably expect to be converted into cash within one year. Current assets include cash and cash equivalents, accounts receivable, inventory, marketable securities, prepaid expenses and other liquid assets that can be readily converted to cash.
# totalcurrentassets: Total assets are the sum of all current and noncurrent assets that a company owns. They are reported on the company balance sheet. The total asset figure is based on the purchase price of the listed assets, and not the fair market value.
# nwc: Working capital, also known as net working capital, is the difference between a company’s current assets, like cash, accounts receivable (customers’ unpaid bills) and inventories of raw materials and finished goods, and its current liabilities, like accounts payable.
# earningsyield: Earnings yield are the earnings per share for the most recent 12-month period divided by the current market price per share. The earnings yield (which is the inverse of the P/E ratio) shows the percentage of each dollar invested in the stock that was earned by the company. The earnings yield is used by many investment managers to determine optimal asset allocations.



class IndicatorsBuilder(object):
    # Targets
    positions = 'positions'

    # Indicators
    p2e = 'pricetoearnings'
    bvps = 'bookvaluespershare'
    shareprice = 't: current_price'
    book2market = 'booktomarket'  # computed as bvps / shareprice TODO check it's correct
    fcf = 'freecashflow'
    totalequity = 'totalequity'
    totalcommonequity = 'totalcommonequity'

    # TODO: use the actual DT to choose which is best, total equity or common only
    fcf2equity = 'fcf2equity'  # computed as fcf / equity TODO: which equity, total or only common
    fcf2commonequity = 'fcf2commonequity'  # computed as fcf / equity TODO: which equity, total or only common

    graham_indicators = ['totalrevenue', 'totalcurrentliabilities',
                         'totalcurrentassets', 'nwc', 'earningsyield',
                         'paymentofdividends',
                         'epsgrowth', p2e, 'pricetobook',
                         'grahamnumber', positions]

    buffet_indicators = [p2e, book2market, fcf2equity, fcf2commonequity,
                         positions]

    # pricetoearnings will be 'nm' (= not meaningful) whenever it is negative
    # TODO maybe it should be substituted by inf, 0, or negative values


    def __init__(self, df, target=positions):
        self.version = 0.1
        self._df = df
        self._target = target

    def get_log(self):
        attrs = self.__dict__.keys()

        attrs.update({'class_code': marshal.dumps(IndicatorsBuilder)})

        return attrs

    # TODO: ask Argimiro what to do with stocks with nan in eps (as Tesla which has no positive earnings)
    def _add_graham_indicators(self):
        def _graham_number(row):
            # sqrt(22.5 * eps * bvps)
            gn = np.NaN
            try:
                gn = sqrt(22.5 * row['basiceps'] * row['bookvaluepershare'])
            except ValueError:
                pass
            return gn

        graham_number = self._df.apply(_graham_number, axis=1)

        self._df = self._df.assign(grahamnumber=graham_number)

    def _add_buffet_indicators(self):
        book2market = self._df.apply(
            lambda row: row[self.bvps] / row[self.shareprice])
        fcf2commonequity = self._df.apply(
            lambda row: row[self.fcf] / row[self.totalcommonequity])
        fcf2equity = self._df.apply(
            lambda row: row[self.fcf] / row[self.totalequity])

        self._df = self._df.assign(book2market=book2market)
        self._df = self._df.assign(fcf2commonequity=fcf2commonequity)
        self._df = self._df.assign(fcf2equity=fcf2equity)

    def to_graham(self):

        self._add_graham_indicators()

        self._df = self._df[self.graham_indicators]

        return self

    def to_buffet(self):

        self._add_buffet_indicators()

        self._df = self._df[self.buffet_indicators]

        return self

    def add_positions(self, threshold):
        def _positions(row):
            if row['t: daily_increase'] >= threshold:
                return 'long'
            elif row['t: daily_increase'] <= -threshold:
                return 'short'
            else:
                return 'neutral'

        positions = self._df.apply(_positions, axis=1)

        self._df = self._df.assign(positions=positions)

        return self

    def build(self):
        return self.Indicators(self._df, self._target)

    # def as_dataset(self, target=positions):
    #     X = self._df.loc[:, self._df.columns != target]
    #     y = self._df[[target]]
    #     return X.values, y.values



    class Indicators(object):

        def __init__(self, df, target):
            self._df = df
            self._target = target

        @property
        def _X_df(self):
            return self._df.loc[:, self._df.columns != self._target]

        @property
        def _y_df(self):
            return self._df[[self._target]]

        @property
        def X(self):
            return self._X_df.values

        @property
        def y(self):
            return self._y_df.values

        @property
        def feature_names(self):
            return self._X_df.columns

        # @property
        # def class_names(self):
        #     return list(set(self._y_df.values.flatten()))

