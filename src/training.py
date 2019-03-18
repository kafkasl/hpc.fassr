import os
import sys
from collections import defaultdict
from time import time

import matplotlib
import numpy as np
import pandas as pd
from pycompss.api.constraint import constraint
from pycompss.api.task import task

from models.classifiers import *
from models.classifiers import train_attrs as attrs
from settings.basic import CACHE_PATH, DATE_FORMAT, CHECKPOINTING
from trading.trade import model_trade, graham_trade, debug_trade
from utils import load_obj, save_obj, dict_to_str

try:
    import pyextrae.multiprocessing as pyextrae

    tracing = True
except:
    tracing = False

matplotlib.use('Agg')

REGRESSION = 'REG'
CLASSIFICATION = 'CAT'


def save(items, filename, path):
    file_path = os.path.join(path, filename)
    print("Saving results %s to %s" % (filename, file_path))

    items.to_csv("%s.csv" % file_path)
    plot = items.plot(title=filename)
    fig = plot.get_figure()
    fig.savefig("%s.png" % file_path)


def get_regression_data(df, attrs, indices, idx, magic_number):
    train = df.loc[indices[idx]:indices[idx + magic_number - 1]]

    test_idx = indices[idx + magic_number]
    if type(test_idx) is not list:
        test_idx = [test_idx]
    test = df.loc[test_idx]

    train_x, train_y = train[attrs], train.y
    test_x, test_y = test[attrs], test.y

    return train_x, train_y, test_x, test_y


def get_classification_data(df, attrs, indices, idx, magic_number):
    train = df.loc[indices[idx]:indices[idx + magic_number - 1]]

    test_idx = indices[idx + magic_number]
    if type(test_idx) is not list:
        test_idx = [test_idx]
    test = df.loc[test_idx]

    train_x, train_y = train[attrs], train.positions
    test_x, test_y = test[attrs], test.positions

    return train_x, train_y, test_x, test_y


def train(df, attrs, clf_class, clf_name, model_params, mode, magic_number,
          dates, dataset_name, trading_params):
    trade_freq = trading_params['trade_frequency']
    name = '%s-%s-attr%s-%s-%s-%s-%s-%s_' % (
        clf_name,
        dataset_name,
        len(attrs),
        dict_to_str(model_params).replace(' ', '_').replace(':', ''),
        mode, magic_number,
        pd.to_datetime(dates[0], format=DATE_FORMAT).date(),
        pd.to_datetime(dates[1], format=DATE_FORMAT).date())
    cached_file = os.path.join(CACHE_PATH, name)

    start_date, final_date = dates
    idx = 0

    indices = sorted(
        [day for day in list(set(df.index.values)) if
         start_date <= day <= final_date])

    print("Model and params: %s %s " % (clf_name, model_params))
    # magic number is by default 53, 52 weeks for training 1 for prediction
    while idx + magic_number < len(indices) and indices[idx + magic_number] <= \
            indices[-1]:

        if mode == CLASSIFICATION:
            train_x, train_y, test_x, test_y = \
                get_classification_data(df, attrs, indices, idx, magic_number)
        elif mode == REGRESSION:
            # get regression datasets (target is float y -> ratio of increase)
            train_x, train_y, test_x, test_y = \
                get_regression_data(df, attrs, indices, idx, magic_number)

        print("Training %s/%s with %s instances." % (
            idx // trade_freq, len(indices) // trade_freq, train_x.shape[0]))
        sys.stdout.flush()

        clf_cached_file = cached_file + str(indices[idx])[:10]

        if not CHECKPOINTING:
            clf = clf_class(**model_params).fit(train_x, train_y)
        else:
            try:
                clf = load_obj(clf_cached_file)
            except:
                clf = clf_class(**model_params).fit(train_x, train_y)
                save_obj(clf, clf_cached_file)

        pred = clf.predict(test_x)

        df.loc[indices[idx + magic_number], clf_name] = pred

        idx += trade_freq

    df_trade = df.dropna(axis=0)

    print("Finished training for %s" % (clf_name))
    return df_trade


@task(returns=2)
def run_regression_model(clf, clf_name, model_params, df, prices, dataset_name,
                         magic_number, mode, trading_params, dates):
    print("Running sequential model %s [%s] with dataset: %s" % (
        clf_name, model_params, dataset_name))
    return run_model(clf, clf_name, model_params, df, prices, dataset_name,
                     magic_number, mode, trading_params, dates)


@task(returns=2)
def run_classification_model(clf, clf_name, model_params, df, prices,
                             dataset_name,
                             magic_number, mode, trading_params, dates):
    print("Running sequential model %s [%s] with dataset: %s" % (
        clf_name, model_params, dataset_name))
    return run_model(clf, clf_name, model_params, df, prices, dataset_name,
                     magic_number, mode, trading_params, dates)


@constraint(ComputingUnits="${ComputingUnits}")
@task(returns=2)
def run_parallel_model(clf, clf_name, model_params, df, prices, dataset_name,
                       magic_number, mode, trading_params, dates):
    global N_JOBS
    print("Running parallel model %s [%s] with dataset: %s" % (
        clf_name, model_params, dataset_name))
    return run_model(clf, clf_name, model_params, df, prices, dataset_name,
                     magic_number, mode, trading_params, dates)


def run_model(clf, clf_name, model_params, df, prices, dataset_name,
              magic_number, mode, trading_params, dates):
    start = time()
    if tracing:
        pro_f = sys.getprofile()
        sys.setprofile(None)

    df_trade = train(df, attrs, clf, clf_name, model_params, mode,
                     magic_number, dates=dates, dataset_name=dataset_name,
                     trading_params=trading_params)

    indices = sorted(
        [day for day in list(set(df_trade.index.values))])
    portfolios = model_trade(df_trade, indices=indices, prices=prices,
                             clf_name=clf_name, trading_params=trading_params,
                             dates=dates)
    total_time = time() - start

    if tracing:
        sys.setprofile(pro_f)

    return (portfolios, total_time)


@task(returns=2)
def run_graham(clf, clf_name, model_params, df, prices, dataset_name,
               magic_number, mode, trading_params, dates):
    start = time()
    if tracing:
        pro_f = sys.getprofile()
        sys.setprofile(None)

    print("Running graham with dataset: %s" % dataset_name)
    start_date = np.datetime64(dates[0])
    final_date = np.datetime64(dates[1])

    indices = sorted(
        [day for day in list(set(df.index.values)) if
         start_date <= day <= final_date])
    # we add the offset of the training data which ain't required for Graham
    indices = indices[magic_number:]
    indices = [indices[i] for i in
               range(0, len(indices), trading_params['trade_frequency'])]

    portfolios = graham_trade(df_trade=df, indices=indices, prices=prices,
                              clf_name=clf_name,
                              trading_params=trading_params, dates=dates)
    total_time = time() - start

    if tracing:
        sys.setprofile(pro_f)

    return (portfolios, total_time)


@task(returns=2)
def run_debug(clf, clf_name, model_params, df, prices, dataset_name,
              magic_number, mode, trading_params, dates):
    start = time()
    if tracing:
        pro_f = sys.getprofile()
        sys.setprofile(None)

    start_date = np.datetime64(dates[0])
    final_date = np.datetime64(dates[1])
    indices = sorted(
        [day for day in list(set(df.index.values)) if
         start_date <= day <= final_date])
    # we add the offset of the training data which ain't required for Graham
    indices = indices[magic_number:]
    indices = [indices[i] for i in
               range(0, len(indices), trading_params['trade_frequency'])]
    print("Running debug with dataset: %s" % dataset_name)

    portfolios = debug_trade(df_trade=df, indices=indices, prices=prices,
                             clf_name=clf_name,
                             trading_params=trading_params, dates=dates)
    total_time = time() - start

    if tracing:
        sys.setprofile(pro_f)

    return (portfolios, total_time)


# TODO: refactor for experiment 2
# @task(returns=dict)
# def run_random(clf, clf_name, model_params, df, prices, dataset_name,
#                magic_number, mode, trading_params):
#     print("Running random with dataset: %s" % dataset_name)
#
#     df_trade = train(df, attrs, clf, clf_name, model_params, mode,
#                      magic_number)
#
#     print("DF trade:")
#     full_print(df_trade)
#     portfolios = random_trade(df_trade, prices=prices,
#                               clf_name=clf_name, trading_params=trading_params)
#     print(get_headers(trading_params=trading_params))
#     print(format_line(dataset_name, clf_name, magic_number,
#                       trading_params,
#                       model_params, portfolios))
#     return portfolios





def explore_models(classifiers, df, prices, dataset_name, save_path,
                   magic_number, trading_params, dates):
    portfolios = defaultdict(list)

    model_jobs = 0
    for clf_name, (clf, model_params) in classifiers.items():
        if 'normal' not in dataset_name and (
                        clf_name == 'random' or clf_name == 'graham'):
            continue

        for params in model_params:

            if clf_name in reg_classifiers.keys():
                mode = REGRESSION
                run = run_regression_model
            elif clf_name in cat_classifiers.keys():
                mode = CLASSIFICATION
                run = run_classification_model
            # elif clf_name == 'random':
            #     mode = REGRESSION
            #     run = run_random
            elif clf_name == 'graham':
                run = run_graham
                mode = REGRESSION
            else:
                run = run_debug
                mode = REGRESSION

            if 'n_jobs' in params.keys():
                run = run_parallel_model

            model_jobs += 1
            res = run(clf, clf_name, params, df, prices, dataset_name,
                      magic_number, mode, trading_params, dates)
            portfolios[clf_name].append((params, trading_params, res))

    return portfolios, model_jobs
