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
    print("Trying to load %s exists? %s" % (
        prices_file, exists_obj(prices_file)))
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
                                      datasets[dataset_name.split(':')[0]][1],
                                      trading_params,
                                      model_params, pfs, total_time))
                params = (dataset_name, clf_name, model_params, trading_params,
                          total_time)
                clean_results.append((params, pfs))
    return clean_results


def get_datasets(period_params, symbols_list_name, thresholds_lst,
                 target_shift,
                 mode='all', datasets=None):
    print("Initializing datasets for periods: %s" % period_params)
    if not datasets:
        datasets = {}
    for thresholds in thresholds_lst:
        for resample_period, magic_number in period_params:
            normal_name, z_name = get_datasets_name(resample_period,
                                                    symbols_list_name,
                                                    thresholds,
                                                    target_shift)

            normal_file = os.path.join(DATA_PATH, normal_name)
            z_file = os.path.join(DATA_PATH, z_name)

            if exists_obj(normal_file) and exists_obj(z_file):
                print("Loading from cache:\n * %s\n * %s" % (
                    normal_file, z_file))
                dfn = load_obj(normal_file)
                dfz = load_obj(z_file)
            else:
                dfn, dfz = get_data(resample_period=resample_period,
                                    symbols_list_name=symbols_list_name,
                                    thresholds=thresholds,
                                    target_shift=target_shift)

            if mode == 'all' or mode == 'normal':
                datasets[normal_name] = (dfn, magic_number, thresholds)
            if mode == 'all' or mode == 'z-score':
                datasets[z_name] = (dfz, magic_number, thresholds)

    return datasets


def get_exp_specific_data(debug: bool, experiment: int):
    if debug:
        return debug_1_classifiers, [(0, 0)]

    if experiment == 1:
        return exp_1_classifiers, [(-np.inf, 0.03), (-np.inf, 0.025),
                                   (-np.inf, 0.02), (-np.inf, 0.015),
                                   (-np.inf, 0.01), (-np.inf, 0.005),
                                   (-np.inf, 0)]

    elif experiment == 2:
        return exp_2_classifiers, [(-0.03, np.inf), (-0.02, np.inf),
                                   (-0.01, np.inf), (0, np.inf),
                                   (-np.inf, 0.03), (-np.inf, 0.02),
                                   (-np.inf, 0.01), (-np.inf, 0),
                                   (-0.03, 0.03),
                                   (-0.02, 0.02), (-0.01, 0.01), (0, 0)]

    elif experiment == 3:
        return exp_3_classifiers, [(-0.03, np.inf), (-0.02, np.inf),
                                   (-0.01, np.inf), (0, np.inf),
                                   (-np.inf, 0.03), (-np.inf, 0.02),
                                   (-np.inf, 0.01), (-np.inf, 0),
                                   (-0.03, 0.03),
                                   (-0.02, 0.02), (-0.01, 0.01), (0, 0)]


def get_trade_mode(trade_mode: str):
    trade_modes = []
    if trade_mode == 'all' or trade_mode == 'sell_all':
        trade_modes.append('sell_all')
    if trade_mode == 'all' or trade_mode == 'avoid_fees':
        trade_modes.append('avoid_fees')
    return trade_modes


if __name__ == '__main__':
    start_time = time()
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default=PROJECT_ROOT)
    parser.add_argument('-s,', '--symbols', type=str, default='sp437')

    parser.add_argument('--experiment', type=int, default=1)
    parser.add_argument('-k', type=int, default=1000,
                        help='Number of top/bot stocks to pick when doing ranking (experiment 3).')
    parser.add_argument('-f', '--trade_frequency', type=int, default=4,
                        help='Number of weeks between each trading session'
                             ' (i.e. 4 = trading monthly)')
    parser.add_argument('--trade_mode', type=str, default='sell_all',
                        choices=['all', 'sell_all', 'avoid_fees'])
    parser.add_argument('--datasets', type=str, default='all',
                        choices=['all', 'normal', 'z-score'])
    parser.add_argument('--start_date', type=str, default='2006-01-01',
                        help="Start date of the period used for training.")
    parser.add_argument('--final_date', type=str, default='2018-06-06',
                        help="Final date of the period used for training.")
    parser.add_argument('--train_period', type=int, default=53,
                        help='Number of weeks to use in each training/predict'
                             'cycle. The last one will be used for testing. 53'
                             'means: 52 for training, predict the 53rd.')

    args = parser.parse_args()

    # Get all inputs and parameters
    start_date = np.datetime64(args.start_date)
    final_date = np.datetime64(args.final_date)
    dates = (start_date, final_date)
    trade_start_date = '2009-03-03'
    trade_final_date = '2018-02-28'
    trade_frequency = args.trade_frequency
    exp = args.experiment
    symbols_list_name = args.symbols
    save_path = args.save_path
    classifiers, thresholds_list = get_exp_specific_data(args.debug, exp)
    trade_modes = get_trade_mode(args.trade_mode)

    # Load prices
    prices = load_prices()

    # Preprocessing
    period_params = [('1W', args.train_period)]
    trade_frequencies = [int(args.trade_frequency)]

    datasets = {}
    for trade_frequency in trade_frequencies:
        datasets = get_datasets(datasets=datasets, period_params=period_params,
                                symbols_list_name=symbols_list_name,
                                thresholds_lst=thresholds_list,
                                mode=args.datasets,
                                target_shift=trade_frequency)

    # Log some execution information for easy access
    print("Models to train: [%s]" % np.sum([len(v[1]) for v in classifiers.values()]))
    pprint(classifiers)
    print("Datasets created [%s] : %s\nArgs: %s" % (len(datasets), datasets.keys(), args))

    results = {}
    total_jobs = 0
    trading_params = {}

    for mode in trade_modes:
        for trade_frequency in trade_frequencies:

            for dataset_name, (df, magic_number, (thresholds)) in datasets.items():
                trading_params = {'k': args.k,
                                  'bot_thresh': thresholds[0],
                                  'top_thresh': thresholds[1],
                                  'mode': mode,
                                  'trade_frequency': trade_frequency,
                                  'dates': (trade_start_date, trade_final_date)}

                results[dataset_name + ':' + mode], jobs = explore_models(
                    classifiers=classifiers, df=df, prices=prices,
                    dataset_name=dataset_name, magic_number=magic_number,
                    save_path=save_path, trading_params=trading_params,
                    dates=dates)
                total_jobs += jobs

    # Log information about the execution
    print(
        "Tasks launched: \n\t* Get data: %s\n\t* Training: %s\n\t* Total: %s" %
        (4 * len(datasets), total_jobs, 1 + 4 * len(datasets) + total_jobs))

    # Print the models performance as the tasks finish.
    print(get_headers(trading_params))
    clean_results = wait_results(results, log=True, datasets=datasets)
    total_time = time()

    # Save the py object containing all Portfolios for each model.
    save_obj(clean_results, os.path.join(save_path, 'clean_results_%s_%s' % (
        symbols_list_name, uuid4().hex[:8])))

    # Print each portfolio per trading session for each model.
    print(clean_results)
    print("Total time: %.3f" % (total_time - start_time))
