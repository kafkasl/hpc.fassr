import os
from collections import defaultdict
from math import sqrt

import matplotlib
import numpy as np
import pandas as pd
from pycompss.api.task import task

from models.classifiers import *
from models.classifiers import train_attrs as attrs
from models.portfolio import Portfolio, Position
from utils import full_print, get_headers, format_line

matplotlib.use('Agg')

REGRESSION = 'REG'
CLASSIFICATION = 'CAT'
final_date = np.datetime64('2018-06-06')


def save(items, filename, path):
    file_path = os.path.join(path, filename)
    print("Saving results %s to %s" % (filename, file_path))

    items.to_csv("%s.csv" % file_path)
    plot = items.plot(title=filename)
    fig = plot.get_figure()
    fig.savefig("%s.png" % file_path)


def get_share_price(prices, day, symbol):
    return prices.loc(axis=0)[(symbol, day)]


def get_k_best(df_trade, day, clf_name, trading_params):
    # sort by predicted increment per stock of classifier clf_name
    df_clf = df_trade[['y', clf_name, 'symbol', 'price']]
    df_aux = df_clf.loc[[day]].sort_values(clf_name)

    k = trading_params['k']
    top_thresh = trading_params['top_thresh']
    bot_thresh = trading_params['bot_thresh']

    botk = df_aux.iloc[0:k].query('%s<%s' % (clf_name, bot_thresh))
    topk = df_aux.iloc[-k - 1: -1].query('%s>%s' % (clf_name, top_thresh))

    return topk, botk


def get_k_random(df_trade, day, clf_name, trading_params):
    df_trade = df_trade[['y', clf_name, 'symbol', 'price']]
    k = trading_params['k']

    sample_size = min(len(df_trade), k)
    topk = df_trade.sample(n=sample_size)
    botk = df_trade.sample(n=sample_size)
    return topk, botk


def graham_screening(df_trade, day, clf_name, trading_params):
    df_y = (df_trade
            .groupby('symbol')
            .resample('1Y')
            .ffill()
            .drop('symbol', axis=1)
            .reset_index()
            .set_index('date')
            .sort_index())

    symbols = list(set(df_trade.symbol))
    screened_symbols = []
    empty_df = pd.DataFrame(columns=df_trade.columns)

    for symbol in symbols:
        df_aux = df_trade.loc[:day + 1].query('symbol=="%s"' % symbol)
        # we have df with only yearly data to compute the last 10/20 years
        # conditions
        df_aux_y = df_y.loc[:day + 1].query('symbol=="%s"' % symbol)

        row_years = df_aux_y.shape[0]

        if row_years == 0:
            continue

        succesful = True

        try:
            day_info = df_aux.loc[day]
        except KeyError:
            print("Day [%s] for symbol %s not found in index." % (day, symbol))
            # full_print(df_aux)
            continue
        succesful &= day_info.revenue > 1500000000
        succesful &= day_info.wc > 0
        succesful &= (df_aux_y.eps > 0).sum() == row_years
        succesful &= (df_aux_y.divpayoutratio > 0).sum() == row_years
        succesful &= day_info.epsgrowth > 0.03

        try:
            pt = sqrt(22.5 * df_aux.loc[day].p2e * df_aux.loc[day].bvps)
            succesful &= df_aux.loc[day].price < pt
        except ValueError as e:
            # A negative graham value is not a problem, just don't square it 
            # print("Negative graham value: %s" % (
            #     22.5 * df_aux.loc[day].p2e * df_aux.loc[day].bvps))
            continue

        if succesful:
            screened_symbols.append(symbol)
    print("Screened symbols: %s" % len(screened_symbols))

    if len(screened_symbols) > 0:
        chosen = df_trade.loc[day].query('symbol in @screened_symbols')
        topk, botk = chosen[['y', 'y', 'symbol', 'price']], empty_df
    else:
        topk, botk = empty_df, empty_df

    return topk, botk


def update_positions(prices, day, available_money, positions, botk, topk):
    new_positions = []
    print("Start %s ============================" % day)
    print("Old positions:\n%s\n" % positions)
    print("Recommended top %s: %s" % (len(topk), list(topk.symbol)))
    print("Recommended bot %s: %s" % (len(botk), list(botk.symbol)))
    for p in positions:
        try:
            current_price = get_share_price(prices, day, p.symbol)
        except Exception as e:
            print("New price for %s not available using last." % p.symbol)
            print(e)
            current_price = p.current_price

        if (p.symbol in botk.symbol.values and p.is_short()) or \
                (p.symbol in topk.symbol.values and p.is_long()):

            # get share price to update the position current_price, other
            # than that the position is exactly the same
            new_position = p.update_price(current_price)

            # we do not pay fees as we already had that stock and position
            new_positions.append(new_position)

            # remove symbol as its already been processed
            topk = topk[topk.symbol != p.symbol]
            botk = botk[botk.symbol != p.symbol]
        else:
            # we sell the positions that we do not continue
            available_money += p.sell(current_price)

    if botk.shape[0] > 0 or topk.shape[0] > 0:
        # initially divide money equally among candidates
        remaining_stocks = (botk.shape[0] + topk.shape[0])

        for (idx, y, pred, symbol, price) in topk.itertuples():
            # get a proportional amount of money to buy
            stash = available_money / remaining_stocks
            # subtract it from the total
            available_money -= stash

            position, extra_money = Position.long(symbol=symbol, price=price,
                                                  available_money=stash)
            new_positions.append(position)

            # extra money of buying only whole shares is returned to the stash
            available_money += extra_money
            remaining_stocks -= 1

        for (idx, y, pred, symbol, price) in botk.itertuples():
            # get a proportional amount of money to buy
            stash = available_money / remaining_stocks
            # subtract it from the total
            available_money -= stash

            position, extra_money = Position.short(symbol=symbol, price=price,
                                                   available_money=stash)
            new_positions.append(position)

            # extra money of buying only whole shares is returned to the stash
            available_money += extra_money
            remaining_stocks -= 1

    print("New positions:\n%s\n" % new_positions)
    print("End %s ============================" % day)

    return available_money, new_positions


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
    # import pdb
    # pdb.set_trace()
    train_x, train_y = train[attrs], train.positions
    test_x, test_y = test[attrs], test.positions

    return train_x, train_y, test_x, test_y


def train(df, attrs, clf, clf_name, model_params, mode, magic_number):
    idx = 0
    indices = sorted(list(set(df.index.values)))

    print("Model params: %s " % model_params)
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
        print(
            "Training %s/%s with %s instances." % (
                idx, len(indices), train_x.shape[0]))

        clf_ = clf(**model_params).fit(train_x, train_y)

        df.loc[indices[idx + magic_number], clf_name] = clf_.predict(test_x)

        idx += 1

    df_trade = df.dropna(axis=0)

    print("Finished training for %s" % (clf_name))
    return df_trade


def _trade(df_trade, prices, clf_name, trading_params, selector_f):
    indices = sorted(
        [day for day in list(set(df_trade.index.values)) if day <= final_date])

    portfolios = []
    positions = []
    stash = 100000

    for day in indices:
        # sort by predicted increment per stock of classifier clf_name

        topk, botk = selector_f(df_trade, day, clf_name, trading_params)

        pf = Portfolio(day, cash=stash, positions=positions)

        portfolios.append(pf)

        old_positions = positions
        old_stash = stash
        stash, positions = update_positions(prices, day, old_stash,
                                            old_positions, botk, topk)
    print("Finished trading for %s" % clf_name)
    return portfolios


def model_trade(df_trade, prices, clf_name, trading_params: dict):
    return _trade(df_trade=df_trade, prices=prices, clf_name=clf_name,
                  trading_params=trading_params, selector_f=get_k_best)


def random_trade(df_trade, prices, clf_name, trading_params: dict):
    indices = sorted(
        [day for day in list(set(df_trade.index.values)) if day <= final_date])
    print("Monkey-Dart trading for %s" % clf_name)

    portfolios = []

    positions = []
    stash = 1000

    for day in indices:
        topk, botk = get_k_random(df_trade, day, clf_name, trading_params)
        print("Random getting topk, botk = %s, %s" % (len(topk), len(botk)))

        pf = Portfolio(day, cash=stash, positions=positions)

        portfolios.append(pf)

        old_positions = positions
        old_stash = stash
        stash, positions = update_positions(prices, day, old_stash,
                                            old_positions, botk, topk)

    return portfolios


def graham_trade(df_trade, prices, clf_name, trading_params):
    return _trade(df_trade=df_trade, prices=prices, clf_name=clf_name,
                  trading_params=trading_params, selector_f=graham_screening)


@task(returns=dict)
def run_model(clf, clf_name, model_params, df, prices, dataset_name,
              magic_number, mode, trading_params):
    print("Running model with dataset: %s" % dataset_name)

    print("DF: %s" % df)

    df_trade = train(df, attrs, clf, clf_name, model_params, mode, magic_number)

    portfolios = model_trade(df_trade, prices=prices,
                             clf_name=clf_name, trading_params=trading_params)

    return portfolios


@task(returns=dict)
def run_random(clf, clf_name, model_params, df, prices, dataset_name,
               magic_number, mode, trading_params):
    print("Running random with dataset: %s" % dataset_name)

    df_trade = train(df, attrs, clf, clf_name, model_params, mode,
                     magic_number)

    print("DF trade:")
    full_print(df_trade)
    portfolios = random_trade(df_trade, prices=prices,
                              clf_name=clf_name, trading_params=trading_params)

    return portfolios


@task(returns=dict)
def run_graham(clf, clf_name, model_params, df, prices, dataset_name,
               magic_number, mode, trading_params):
    print("Running graham with dataset: %s" % dataset_name)

    print("Columns: %s" % df.columns)

    res = graham_trade(df_trade=df, prices=prices, clf_name=clf_name,
                       trading_params=trading_params)

    print(get_headers(trading_params=trading_params))
    print(format_line(dataset_name, clf, magic_number, trading_params,
                      model_params, res))
    return res


def explore_models(classifiers, df, prices, dataset_name, save_path,
                   magic_number, trading_params):
    portfolios = defaultdict(list)

    model_jobs = 0
    for clf_name, (clf, model_params) in classifiers.items():
        if 'normal' not in dataset_name and (
                        clf_name == 'random' or clf_name == 'graham'):
            continue

        if clf_name in reg_classifiers.keys():
            mode = REGRESSION
            run = run_model
        elif clf_name in cat_classifiers.keys():
            mode = CLASSIFICATION
            run = run_model
        elif clf_name == 'random':
            mode = REGRESSION
            run = run_random
        else:
            run = run_graham
            mode = REGRESSION

        for params in model_params:
            model_jobs += 1
            res = run(clf, clf_name, params, df, prices, dataset_name,
                      magic_number, mode, trading_params)
            portfolios[clf_name].append((params, trading_params, res))

    return portfolios, model_jobs
