import argparse
import json
import os
from time import time

from pycompss.api.api import compss_wait_on

from data_managers.data_collector import get_data, get_prices
from models.classifiers import *
from settings.basic import PROJECT_ROOT
from training import explore_models
from utils import save_obj


def get_trading_params(ks):
    params = []
    threshold = 0.03
    for k in ks:
        params.append({'k': k,
                       'bot_thresh': threshold,
                       'top_thresh': threshold})

    return params


if __name__ == '__main__':
    start_time = time()
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default=PROJECT_ROOT)

    args = parser.parse_args()

    # datasets = ['normal', 'z-score']
    resample_periods = ['3M']
    symbols_list_name = 'dow30'
    magic_number = 13
    trading_params = [{'k': 10, 'bot_thresh': 0, 'top_thresh': 0}]
    classifiers = debug_classifiers
    save_path = args.save_path

    if not args.debug:
        resample_periods = ['1W', '2W', '1M', '3M']
        symbols_list_name = 'sp500'
        magic_number = 53
        trading_params = get_trading_params([10])
        classifiers = debug_classifiers

    results = {}
    accuracies = {}
    prices = get_prices(symbols_list_name='sp500',
                        resample_period='1D', only_prices=True)

    for resample_period in resample_periods:
        dfn, dfz, attrs = get_data(resample_period=resample_period,
                                   symbols_list_name=symbols_list_name)

        datasets = {'normal': dfn, 'z-score': dfz}

        for d_key in datasets.keys():
            for params in trading_params:
                df = datasets[d_key]

                dataset_name = '%s_%s' % (d_key, resample_period)
                portfolios = explore_models(classifiers=classifiers,
                                            df=df, prices=prices,
                                            dataset_name=dataset_name,
                                            attrs=attrs,
                                            magic_number=magic_number,
                                            save_path=save_path,
                                            trading_params=params)

                results[dataset_name] = portfolios

    print("All training tasks launched, proceeding to wait for the results.")
    for dataset_name, portfolios in results.items():
        # wait on everything
        for clf_name in portfolios.keys():
            for i in range(0, len(portfolios[clf_name])):
                model_params, trading_params, res = portfolios[clf_name][i]
                res = compss_wait_on(res)
                portfolios[clf_name][i] = (model_params, trading_params, res)


    for dataset_name, res in results.items():
        for clf, lst in res.items():
            for clf_params, trading_params, pfs in lst:
                print('%s,%s,%s,%s,%.3f' % (
                    dataset_name, clf, json.dumps(clf_params),
                    json.dumps(trading_params),
                    pfs[-1].total_money))

    save_obj(results,
             os.path.join(save_path, 'full_results_%s' % symbols_list_name))
    print("Waiting for the results.")
    print(results)
    total_time = time()
    print("Total time: %.3f" % (total_time - start_time))
