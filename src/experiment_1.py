import argparse
import os
from pprint import pprint
from time import time
from uuid import uuid4

import numpy as np
from pycompss.api.api import compss_wait_on

from data_managers.data_collector import get_data, get_prices
from models.classifiers import *
from settings.basic import PROJECT_ROOT, DATA_PATH
from training import explore_models
from utils import save_obj, get_headers, format_line, load_obj, exists_obj, \
    get_datasets_name


def load_prices(prices_list='sp500'):
    prices_file = os.path.join(DATA_PATH, 'prices_' + prices_list)
    print("Trying to load %s exists? %s" %(prices_file, exists_obj(prices_file)))
    if exists_obj(prices_file):
        print("Loading from cache:\n * %s" % prices_file)
        prices = load_obj(prices_file)
    else:
        prices = get_prices(symbols_list_name=prices_list,
                            resample_period='1D', only_prices=True)

    return prices


def wait_results(results, log=False, datasets=None):
    clean_results = []
    for dataset_name, portfolios in results.items():
        # wait on everything
        for clf_name in portfolios.keys():
            for i in range(0, len(portfolios[clf_name])):
                model_params, trading_params, res = portfolios[clf_name][i]
                res = compss_wait_on(res)
                pfs, total_time = res
                portfolios[clf_name][i] = (model_params, trading_params, res)
                if log:
                    print(format_line(dataset_name, clf_name,
                                      datasets[dataset_name][1],
                                      trading_params,
                                      model_params, pfs, total_time))
                params = (dataset_name, clf_name, model_params, trading_params,
                          total_time)
                clean_results = (params, pfs)
    return clean_results


def get_datasets(period_params, symbols_list_name, thresholds, target_shift,
                 mode='all'):
    print("Initializing datasets for periods: %s" % period_params)
    datasets = {}
    for resample_period, magic_number in period_params:
        normal_name, z_name = get_datasets_name(resample_period,
                                                symbols_list_name, thresholds,
                                                target_shift)

        normal_file = os.path.join(DATA_PATH, normal_name)
        z_file = os.path.join(DATA_PATH, z_name)

        if exists_obj(normal_file) and exists_obj(z_file):
            print("Loading from cache:\n * %s\n * %s" % (normal_file, z_file))
            dfn = load_obj(normal_file)
            dfz = load_obj(z_file)
        else:
            dfn, dfz = get_data(resample_period=resample_period,
                                symbols_list_name=symbols_list_name,
                                thresholds=thresholds,
                                target_shift=target_shift)

        if mode == 'all' or mode == 'normal':
            datasets[normal_name] = (dfn, magic_number)
        if mode == 'all' or mode == 'z-score':
            datasets[z_name] = (dfz, magic_number)

    return datasets


if __name__ == '__main__':
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
    parser.add_argument('--experiment', type=int, default=1)
    parser.add_argument('--trade_frequency', type=int, default=4,
                        help='Number of weeks between each trading session'
                             ' (i.e. 4 = trading monthly)')
    parser.add_argument('--trade_mode', type=str, default='sell_all',
                        choices=['sell_all', 'avoid_fees'])
    parser.add_argument('--datasets', type=str, default='all',
                        choices=['all', 'normal', 'z-score'])
    parser.add_argument('--start_date', type=str, default='2006-01-01')
    parser.add_argument('--final_date', type=str, default='2018-06-06')
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

    trade_frequency = args.trade_frequency
    bot_threshold = args.bot_threshold if args.bot_threshold > -66 else -np.inf
    top_threshold = args.top_threshold if args.top_threshold < 66 else np.inf
    thresholds = (bot_threshold, top_threshold)

    experiment = args.experiment
    symbols_list_name = args.symbols

    trading_params = {'k': 1000,
                      'bot_thresh': bot_threshold,
                      'top_thresh': top_threshold,
                      'mode': args.trade_mode,
                      'trade_frequency': trade_frequency,
                      'dates': (trade_start_date, trade_final_date)}

    period_params = [('1W', args.train_period)]
    classifiers = debug_1_classifiers
    save_path = args.save_path

    if not args.debug:

        # period_params = [('1W', 53), ('1SM', 27), ('1BM', 14), ('1Q', 7)]
        if experiment == 1:
            classifiers = experiment_1_classifiers
        elif experiment == 2:
            classifiers = experiment_2_classifiers

    results = {}
    prices_file = os.path.join(DATA_PATH, 'prices_sp500')
    if exists_obj(prices_file):
        print("Loading from cache:\n * %s" % prices_file)
        prices = load_obj(prices_file)
    else:
        prices = get_prices(symbols_list_name='sp500',
                            resample_period='1D', only_prices=True)
        # save_obj(prices, prices_file)

    datasets = get_datasets(period_params=period_params,
                            symbols_list_name=symbols_list_name,
                            thresholds=thresholds,
                            mode=args.datasets,
                            target_shift=trade_frequency)

    # Log some execution information for easy access
    print("Models to train:")
    pprint(classifiers)
    print("Datasets created: %s" % (datasets.keys()))
    print("Args: %s" % args)

    total_jobs = 0
    for dataset_name, (df, magic_number) in datasets.items():
        results[dataset_name], jobs = explore_models(classifiers=classifiers,
                                                     df=df, prices=prices,
                                                     dataset_name=dataset_name,
                                                     magic_number=magic_number,
                                                     save_path=save_path,
                                                     trading_params=trading_params,
                                                     dates=dates)
        total_jobs += jobs

    print("%s get data tasks launched, proceeding to: wait for the results." %
          (4 * len(datasets)))
    print(
        "%s training tasks launched, proceeding to: wait for the results." % total_jobs)
    print("Total expected jobs: %s" % (1 + 4 * len(datasets) + total_jobs))

    print(get_headers(trading_params))
    clean_results = wait_results(results, log=True, datasets=datasets)

    save_obj(clean_results,
             os.path.join(save_path, 'clean_results_%s_%s' % (
                 symbols_list_name, uuid4().hex[:8])))

    print(clean_results)
    total_time = time()
    print("Total time: %.3f" % (total_time - start_time))
