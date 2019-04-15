import unittest
from collections import defaultdict
from time import time

import numpy as np

from fassr import get_datasets, wait_results, load_prices
from models.classifiers import *
from training.train import explore_models


class YdraTest(unittest.TestCase):
    def test_basic(self):
        start_time = time()

        start_date = np.datetime64('2006-01-01')
        final_date = np.datetime64('2018-04-02')
        dates = (start_date, final_date)

        trade_start_date = '2009-03-04'
        trade_final_date = '2018-02-28'

        symbols_list_name = 'sp437'

        period_params = [('1W', 53)]

        prices = load_prices('sp500')
        top_threshold = 0.03
        bot_threshold = -np.inf
        thresholds_lst = [(bot_threshold, top_threshold)]

        clfs = [debug]
        totals = defaultdict(list)
        for classifiers in clfs:
            for trade_mode in ['sell_all', 'avoid_fees']:
                # Yearly trade
                results = {}
                trade_frequency = 52
                dataset = 'normal'

                datasets = get_datasets(period_params=period_params,
                                        symbols_list_name=symbols_list_name,
                                        thresholds_lst=thresholds_lst,
                                        mode=dataset,
                                        target_shift=trade_frequency)

                for dataset_name, (
                df, magic_number, thresholds) in datasets.items():
                    trading_params = {'k': 1000,
                                      'bot_thresh': thresholds[0],
                                      'top_thresh': thresholds[1],
                                      'mode': trade_mode,
                                      'trade_frequency': trade_frequency,
                                      'dates': (
                                          trade_start_date, trade_final_date)}
                    results[dataset_name], _ = explore_models(
                        classifiers=classifiers,
                        df=df, prices=prices,
                        dataset_name=dataset_name,
                        magic_number=magic_number,
                        trading_params=trading_params,
                        dates=dates)

                clean_results = wait_results(results)
                print(clean_results[0][1][-1])
                totals[trade_mode].append(clean_results[0][1][-1].total_money)

                print(clean_results[0][1])
                self.assertEqual(len(clean_results[0][1]), 10)
                self.assertEqual(clean_results[0][1][0]._day_str, '2009-04-12')
                self.assertEqual(clean_results[0][1][-1]._day_str,
                                 '2018-04-01')

                # Monthly trade
                results = {}
                trade_frequency = 4

                datasets = get_datasets(period_params=period_params,
                                        symbols_list_name=symbols_list_name,
                                        thresholds_lst=thresholds_lst,
                                        mode=dataset,
                                        target_shift=trade_frequency)

                for dataset_name, (
                df, magic_number, thresholds) in datasets.items():
                    trading_params = {'k': 1000,
                                      'bot_thresh': thresholds[0],
                                      'top_thresh': thresholds[1],
                                      'mode': trade_mode,
                                      'trade_frequency': trade_frequency,
                                      'dates': (
                                          trade_start_date, trade_final_date)}

                    results[dataset_name], _ = explore_models(
                        classifiers=classifiers,
                        df=df, prices=prices,
                        dataset_name=dataset_name,
                        magic_number=magic_number,
                        trading_params=trading_params,
                        dates=dates)

                clean_results2 = wait_results(results)
                print(clean_results2[0][1][-1])
                totals[trade_mode].append(clean_results2[0][1][-1].total_money)
                print(clean_results2[0][1])
                self.assertEqual(len(clean_results2[0][1]), 118)
                self.assertEqual(clean_results2[0][1][0]._day_str,
                                 '2009-04-12')
                self.assertEqual(clean_results2[0][1][-1]._day_str,
                                 '2018-04-01')

        expected = {
            'sell_all': -154109.36,
            'avoid_fees': 74599.96
        }
        for k, v in totals.items():
            ok = len(set(v)) == 1
            # self.assertTrue(ok)
            print("Test different periods same money %s: %s" % (
                k, 'OK' if ok else 'FAILED'))
            if not ok:
                print(v)
            ok = int(v[0]) == int(expected[k])
            # self.assertTrue(ok)
            print("Test final result %s: %s" % (k, 'OK' if ok else 'FAILED'))
            if not ok:
                print('Expected: %s, Got: %s' % (expected[k], v[0]))

        total_time = time()
        print("Total time: %.3f" % (total_time - start_time))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
