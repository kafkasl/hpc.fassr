import argparse
from collections import defaultdict
from time import time

import numpy as np

from experiment_1 import get_datasets, wait_results, load_prices
from models.classifiers import *
from settings.basic import PROJECT_ROOT
from training import explore_models
import unittest

class YdraTest(unittest.TestCase):
    def test_basic(self):
        start_time = time()
        parser = argparse.ArgumentParser()

        parser.add_argument('-d', '--debug', action='store_true', default=False)
        parser.add_argument('--save_path', type=str, default=PROJECT_ROOT)
        parser.add_argument('-s,', '--symbols', type=str, default='sp437')
        parser.add_argument('--bot_threshold', type=float, default=-0.015,
                            help='Bottom threshold used to create '
                                 'categorical y labels, and trading'
                                 ' threshold in ranking. -66 indicates'
                                 ' a -np.inf threshold')
        parser.add_argument('--top_threshold', type=float, default=0.015,
                            help='Top threshold used to create '
                                 'categorical y labels, and trading'
                                 ' threshold in ranking. 66 indicates'
                                 ' a np.inf threshold'
                            )
        parser.add_argument('--trade_frequency', type=int, default=4,
                            help='Number of weeks between each trading session'
                                 ' (i.e. 4 = trading monthly)')
        parser.add_argument('--trade_mode', type=str, default='sell_all',
                            choices=['sell_all', 'avoid_fees'])
        parser.add_argument('--datasets', type=str, default='all',
                            choices=['all', 'normal', 'z-score'])
        parser.add_argument('--start_date', type=str, default='2006-01-01')
        parser.add_argument('--final_date', type=str, default='2018-04-02')
        parser.add_argument('--train_period', type=int, default=53,
                            help='Number of weeks to use in each training/predict'
                                 'cycle. The last one will be used for testing. 53'
                                 'means: 52 for training, predict the 53rd.')

        args = parser.parse_args()

        start_date = np.datetime64(args.start_date)
        final_date = np.datetime64(args.final_date)
        dates = (start_date, final_date)

        trade_start_date = '2009-03-04'
        trade_final_date = '2018-02-28'

        symbols_list_name = 'sp437'

        period_params = [('1W', args.train_period)]
        save_path = args.save_path

        prices = load_prices('sp500')
        top_threshold = 0.03
        bot_threshold = -np.inf
        thresholds = (bot_threshold, top_threshold)
        clfs = [{'SVC': (SVC, [{'C': 0.125, 'gamma': 0.125}])}, graham, debug]
        totals = defaultdict(list)
        for classifiers in clfs:
            for trade_mode in ['sell_all', 'avoid_fees']:
                # Yearly trade
                results = {}
                trade_frequency = 52
                dataset = 'normal'

                datasets = get_datasets(period_params=period_params,
                                        symbols_list_name=symbols_list_name,
                                        thresholds=thresholds,
                                        mode=dataset,
                                        target_shift=trade_frequency)

                trading_params = {'k': 1000,
                                  'bot_thresh': bot_threshold,
                                  'top_thresh': top_threshold,
                                  'mode': trade_mode,
                                  'trade_frequency': trade_frequency,
                                  'dates': (trade_start_date, trade_final_date)}

                for dataset_name, (df, magic_number) in datasets.items():
                    results[dataset_name], _ = explore_models(classifiers=classifiers,
                                                              df=df, prices=prices,
                                                              dataset_name=dataset_name,
                                                              magic_number=magic_number,
                                                              save_path=save_path,
                                                              trading_params=trading_params,
                                                              dates=dates)

                clean_results = wait_results(results)
                print(clean_results[0][1][-1])
                totals[trade_mode].append(clean_results[0][1][-1].total_money)

                print(clean_results[0][1])
                self.assertEqual(len(clean_results[0][1]), 10)
                self.assertEqual(clean_results[0][1][0]._day_str, '2009-04-12')
                self.assertEqual(clean_results[0][1][-1]._day_str, '2018-04-01')


                # Monthly trade
                results = {}
                trade_frequency = 4

                datasets = get_datasets(period_params=period_params,
                                        symbols_list_name=symbols_list_name,
                                        thresholds=thresholds,
                                        mode=dataset,
                                        target_shift=trade_frequency)

                trading_params = {'k': 1000,
                                  'bot_thresh': bot_threshold,
                                  'top_thresh': top_threshold,
                                  'mode': trade_mode,
                                  'trade_frequency': trade_frequency,
                                  'dates': (trade_start_date, trade_final_date)}

                for dataset_name, (df, magic_number) in datasets.items():
                    results[dataset_name], _ = explore_models(classifiers=classifiers,
                                                              df=df, prices=prices,
                                                              dataset_name=dataset_name,
                                                              magic_number=magic_number,
                                                              save_path=save_path,
                                                              trading_params=trading_params,
                                                              dates=dates)

                clean_results2 = wait_results(results)
                print(clean_results2[0][1][-1])
                totals[trade_mode].append(clean_results2[0][1][-1].total_money)
                print(clean_results2[0][1])
                self.assertEqual(len(clean_results2[0][1]), 118)
                self.assertEqual(clean_results2[0][1][0]._day_str, '2009-04-12')
                self.assertEqual(clean_results2[0][1][-1]._day_str, '2018-04-01')


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
            ok = v[0] == expected[k]
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


