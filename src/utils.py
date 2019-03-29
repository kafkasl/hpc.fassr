import base64
import gzip
import json
import os
import pickle
from glob import glob
from urllib.parse import urlparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

from settings.basic import (CACHE_ENABLED, CACHE_PATH, DATA_PATH,
                            intrinio_username,
                            intrinio_password, debug)


def dict_to_str(dct):
    return ' '.join(['%s:%s' % (k, v) for k, v in dct.items()])


def get_datasets_name(resample_period, symbols_list_name, thresholds,
                      target_shift):
    normal_name = "normal_%s_%s_%s_%s_y%s" % (
        resample_period, symbols_list_name, thresholds[0],
        thresholds[1],
        target_shift)
    z_name = "z-score_%s_%s_%s_%s_y%s" % (
        resample_period, symbols_list_name, thresholds[0], thresholds[1],
        target_shift)
    return normal_name, z_name


def get_headers(trading_params):
    header = 'dataset,period,clf,magic,model_params,'
    header += ','.join(
        [k for k in trading_params.keys() if k != 'dates'])
    header += ',start_trade,final_trade,time,min,max,mean,last'

    return header


def format_line(dataset_name, clf, magic, trading_params, model_params, pfs,
                total_time):
    r = [p.total_money for p in pfs]
    line = '%s,%s,%s,%s,%s,' % (
        dataset_name.split('_')[0], dataset_name.split('_')[1], clf, magic,
        dict_to_str(model_params))
    line += ','.join(list([str(v) for v in trading_params.values()])[:-1])
    line += ',' + trading_params['dates'][0] + ',' + \
            trading_params['dates'][1] + ','
    line += '%.2f,' % total_time
    line += '%.1f,%.1f,%.1f,%.1f' % (np.min(r), np.max(r), np.mean(r), r[-1])

    return line


def full_print(res):
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):
        print(res)


def exists_obj(name):
    return os.path.exists(name + '.pgz')


def save_obj(obj, name):
    with gzip.GzipFile(name + '.pgz', 'w') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with gzip.GzipFile(name + '.pgz', 'r') as f:
        return pickle.load(f)


def to_df(file: str) -> pd.DataFrame:
    df = pd.read_csv(file)

    df.set_index(['year', 'quarter'], inplace=True)
    df.sort_index(inplace=True)

    return df


def plot(x, y):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.plot(x, y)
    plt.gcf().autofmt_xdate()


def load_symbol_list(symbols_list_name: str) -> list:
    path = os.path.join(DATA_PATH, '%s_symbols.lst' % (symbols_list_name))
    return open(path).read().split()


def call_and_cache(url: str, cache=True) -> dict:
    """
    Calls the URL with GET method if the url file is not cached
    :param url: url to retrieve
    :param kwargs: specify no-cache
    :return: json.loads of the response (or empty dict if error)
    """
    url_parsed = urlparse(url)

    cached_file = os.path.join(CACHE_PATH,
                               url_parsed.netloc + url_parsed.path + "/" +
                               base64.standard_b64encode(
                                   url_parsed.query.encode()).decode())

    if not os.path.exists(os.path.dirname(cached_file)):
        os.makedirs(os.path.dirname(cached_file))

    data_json = {}
    if CACHE_ENABLED and os.path.exists(cached_file) and cache:
        if debug:
            print(
                "Data was present in cache and cache is enabled, loading: %s for %s" %
                (cached_file, url))
        with open(cached_file, 'r') as f:
            data_json = json.loads(f.read())
    else:
        print(
            "Data was either not present in cache or it was disabled calling request: %s" % url)
        r = requests.get(url, auth=HTTPBasicAuth(intrinio_username,
                                                 intrinio_password))

        if r.status_code != 200:
            print(
                "Request status was: %s for URL: %s" % (r.status_code, url))
            return data_json

        data_json = json.loads(r.text)

        if 'data' in data_json.keys() and not len(data_json['data']) > 0:
            print("Data field is empty.\nRequest URL: %s" % (url))

        with open(cached_file, 'w') as f:
            f.write(json.dumps(data_json))
            print(
                "Successfully cached url: %s to %s" % (url, cached_file))

    return data_json


def plot_2_axis():
    import numpy as np
    import matplotlib.pyplot as plt

    x, y = np.random.random((2, 50))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter(df['last'], df.C, c='b')
    ax2.scatter(df['last'], df.gamma, c='r')
    ax1.set_yscale('log')
    ax2.set_yscale('log')



def load_trend(file, price, name):
    res = load_obj(file)
    res = [r for r in res if '%.1f' % r[1][-1].total_money == str(price)]
    return to_df_col(res[0][1], name)


def get_trend(results, price, name):
    res = []
    # from glob import glob
    # for file in glob('*/clean_results_*'):
    #     import ipdb
    #     ipdb.set_trace()
    # result = load_obj(file[:-4])
    for result in results:
        res.extend(
            [r for r in result if '%.1f' % r[1][-1].total_money == str(price)])
    # del result
    if len(res) == 0:
        print("No results found")
        return
    # break
    return to_df_col(res[0][1], name)


def load_all(experiment):
    from glob import glob
    results = []
    for file in glob('%s/*/clean_results_*' % experiment):
        print("loading: %s" % file)
        results.append(load_obj(file[:-4]))

    return results


def new_plot(file, experiment):
    cols = ["dataset", "period", "clf", "magic", "model_params", "k",
            "bot_thresh", "top_thresh", "mode", "trade_frequency",
            "start_trade", "final_trade", "time", "min", "max", "mean", "last"]

    results = load_all(experiment)
    r1 = pd.read_csv(file, names=cols).sort_values('last').drop('time',
                                                                1).drop_duplicates()
    best = r1.groupby('clf')[['last']].max()

    sp500 = pd.read_csv('sp500.csv').set_index('Date')
    sp500.index = pd.to_datetime(sp500.index)
    sp500 = sp500[['Adj Close']].rename(columns={'Adj Close': 'S&P 500'})
    ratio = 100000 / sp500.iloc[0]

    trends = []
    names = ['AdaBoost', 'NN', 'RF', 'SVM', 'Graham', 'S&P 500']
    for i, (clf, price) in enumerate(best.itertuples()):
        trends.append(get_trend(results, price, names[i]))

    sptrend = sp500 * ratio
    sptrend = sptrend.resample('1W').last()
    sptrend = sptrend[sptrend.index.isin(trends[0].index)]
    trends.append(sptrend)
    df = pd.concat(trends, axis=1).interpolate()

    df.index = pd.to_datetime(df.index)

    return df





def plot_scp():
    mlpc = load_trend('clean_results_sp437_2bb95299', 1413316.4, 'NN')
    svc = load_trend('clean_results_sp437_2bb95299', 1317296.2, 'SVC')
    rfc = load_trend('clean_results_sp437_1feda273', 629870.5, 'RFC')
    adaboost = load_trend('clean_results_sp437_41cb0e58', 620100.8, 'AdaBoost')
    graham = load_trend('clean_results_sp437_2bb95299', 547199.6, 'Graham')

    df = pd.concat([mlpc, svc, rfc, adaboost, graham], axis=1).interpolate()
    df.index = pd.to_datetime(df.index)
    return df
