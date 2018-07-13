from data_managers.intrinio import *
from datetime import datetime
from settings.basic import DATE_FORMAT

import unittest


class SampleValues(unittest.TestCase):
    tags = ['totalrevenue', 'totalassets', 'totalliabilities', 'basiceps', 'paymentofdividends',
            'pricetoearnings', 'bookvaluepershare']

    def test_tag_single_day(self):
        symbol = 'AAPL'
        date = datetime(year=2018, month=3, day=31)

        for tag in SampleValues.tags:
            data_json = get_tag(symbol=symbol, tag=tag, start_date=date, end_date=date)

            print("Got data for %s and symbol %s for day %s" % (tag, symbol, date))
            # # print(data_json)
            # self.assertEqual(data_json['data'][0]['value'], 247417000000.0)
            # self.assertEqual(data_json['data'][0]['date'], date.strftime(DATE_FORMAT))
            # self.assertEqual(data_json['identifier'], symbol)
            # self.assertEqual(data_json['item'], 'totalrevenue')

    def test_tag_range(self):
        symbol = 'AAPL'
        start_date = datetime(year=2000, month=3, day=31)
        end_date = datetime(year=2018, month=3, day=31)

        for tag in SampleValues.tags:

            try:
                data_json = get_tag(symbol=symbol, tag=tag, start_date=start_date,
                                    end_date=end_date)
                print("Got data for %s and symbol %s for range %s - %s" %
                      (tag, symbol, start_date, end_date))
                # This raises exception, not all data tags return same number of points
                print(len(data_json['data']))
                # self.assertEqual(len(data_json['data']), 37)
            except Exception as e:
                raise Exception("ERROR: Excetion %s for %s and symbol %s for range %s - %s" %
                                (e, tag, symbol, start_date, end_date))
                # print(data_json)
                # for i, item in enumerate(data_json['data']):
                #     print("Item %s: %s" % (i, item))
                # self.assertEqual(data_json['data'][0]['value'], 247417000000.0)
                # self.assertEqual(data_json['data'][0]['date'], start_date.strftime(DATE_FORMAT))
                # self.assertEqual(data_json['identifier'], symbol)
                # self.assertEqual(data_json['item'], 'totalrevenue')


    def test_graham_df(self):
        symbol = 'AAPL'
        start_date = datetime(year=2000, month=3, day=31)
        end_date = datetime(year=2018, month=3, day=31)

        df = build_df_for_graham(symbol=symbol, start_date=start_date, end_date=end_date)

        print(df)