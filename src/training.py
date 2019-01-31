import os
from collections import defaultdict
from math import sqrt

import matplotlib
import numpy as np
import pandas as pd
from pycompss.api.task import task

from models.classifiers import *
from models.portfolio import Portfolio, Position
from utils import full_print

matplotlib.use('Agg')

REGRESSION = 'REG'
CLASSIFICATION = 'CAT'
final_date = np.datetime64('2018-10-18')


def save(items, filename, path):
    file_path = os.path.join(path, filename)
    print("Saving results %s to %s" % (filename, file_path))

    items.to_csv("%s.csv" % file_path)
    plot = items.plot(title=filename)
    fig = plot.get_figure()
    fig.savefig("%s.png" % file_path)


def get_share_price(prices, day, symbol):
    try:
        price = prices.loc(axis=0)[(symbol, day)]
        return price
    except Exception as e:
        print("Exception: [%s] for key (%s, %s)" % (e, day, symbol))
        print("Prices:")
        full_print(prices)
        raise Exception(e)


def get_k_best(df_aux, clf_name, trading_params):
    k = trading_params['k']
    top_thresh = trading_params['top_thresh']
    bot_thresh = trading_params['bot_thresh']

    botk = df_aux.iloc[0:k].query('%s<%s' % (clf_name, bot_thresh))
    topk = df_aux.iloc[-k - 1: -1].query('%s>%s' % (clf_name, top_thresh))

    return topk, botk


def get_k_random(df_trade, k):

    sample_size = min(len(df_trade), k)
    topk = df_trade.sample(n=sample_size)
    botk = df_trade.sample(n=sample_size)
    return topk, botk


def graham_screening(df, day, k):
    df_y = (df
            .groupby('symbol')
            .resample('1Y')
            .ffill()
            .drop('symbol', axis=1)
            .reset_index()
            .set_index('date')
            .sort_index())

    symbols = list(set(df.symbol))
    screened_symbols = []
    empty_df = pd.DataFrame(columns=df.columns)

    for symbol in symbols:
        df_aux = df.loc[:day+1].query('symbol=="%s"' % symbol)
        # we have df with only yearly data to compute the last 10/20 years
        # conditions
        df_aux_y = df_y.loc[:day+1].query('symbol=="%s"' % symbol)

        row_years = df_aux_y.shape[0]

        if row_years == 0:
            continue

        succesful = True

        try:
            day_info = df_aux.loc[day]
        except KeyError as e:
            print("Day [%s] for symbol %s not found in indices: %s" % (day, symbol, df.query('symbol=="%s"' % symbol).index.values))
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
            print("Negative graham value: %s" % (
                22.5 * df_aux.loc[day].p2e * df_aux.loc[day].bvps))
            continue

        if succesful:
            print("Screened symbols: %s" % screened_symbols)
            screened_symbols.append(symbol)

    if len(screened_symbols) > 0:
        chosen = df.loc[day].query('symbol in @screened_symbols')
        sample_size = min(2*k, len(chosen))
        chosen = chosen.sample(n=sample_size)
        # if len(screened_symbols) > 2 * k:
        #     chosen = df[df.symbol.isin(screened_symbols)].sample(n=2 * k)
        # else:
        #     chosen = df[df.symbol.isin(screened_symbols)].sample(
        #         n=len(screened_symbols))

        topk, botk = chosen[['y', 'y', 'symbol']], empty_df
    else:
        topk, botk = empty_df, empty_df

    return topk, botk


def update_positions(prices, day, available_money, positions, botk, topk):
    new_positions = []
    for p in positions:
        current_price = get_share_price(prices, day, p.symbol)

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

        for (idx, y, pred, symbol) in topk.itertuples():
            price = get_share_price(prices, day, symbol)
            # get a proportional amount of money to buy
            stash = available_money / remaining_stocks
            # subtract it from the total
            available_money -= stash

            position, extra_money = Position.long(symbol=symbol, price=price,
                                                  available_money=stash)
            new_positions.append(position)

            # the extra money of buying only whole shares is returned to the stash
            available_money += extra_money
            remaining_stocks -= 1

        for (idx, y, pred, symbol) in botk.itertuples():
            price = get_share_price(prices, day, symbol)
            # get a proportional amount of money to buy
            stash = available_money / remaining_stocks
            # subtract it from the total
            available_money -= stash

            # shorting spends always all the investment
            position, _ = Position.short(symbol=symbol, price=price,
                                         available_money=stash)
            new_positions.append(position)

            remaining_stocks -= 1

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


def train(df, attrs, clf, clf_name, models_params, mode, magic_number):
    idx = 0
    indices = sorted(list(set(df.index.values)))

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

        clf_ = clf(**models_params).fit(train_x, train_y)

        df.loc[indices[idx + magic_number], clf_name] = clf_.predict(test_x)

        idx += 1

    df_trade = df.dropna(axis=0)

    print("Finished training for %s" % (clf_name))
    return df_trade


def trade(df_trade, prices, clf_name, trading_params: dict):
    indices = sorted(
        [day for day in list(set(df_trade.index.values)) if day <= final_date])
    df_clf = df_trade[['y', clf_name, 'symbol']]
    print("Beginning trading for %s" % clf_name)

    portfolios = []

    positions = []
    stash = 1000

    for day in indices:
        # sort by predicted increment per stock of classifier clf_name
        df_aux = df_clf.loc[[day]].sort_values(clf_name)

        topk, botk = get_k_best(df_aux, clf_name, trading_params)

        pf = Portfolio(day, cash=stash, positions=positions)

        portfolios.append(pf)

        old_positions = positions
        old_stash = stash
        stash, positions = update_positions(prices, day, old_stash,
                                            old_positions,
                                            botk, topk)
    print("Finished trading for %s" % clf_name)
    return portfolios


def random_trade(df_trade, prices, clf_name, trading_params: dict):
    k = trading_params['k']

    indices = sorted(
        [day for day in list(set(df_trade.index.values)) if day <= final_date])
    print("Monkey-Dart trading for %s" % clf_name)

    portfolios = []

    positions = []
    stash = 1000

    for day in indices:
        topk, botk = get_k_random(df_trade, k)
        print("Random getting topk, botk = %s, %s" % (len(topk), len(botk)))

        pf = Portfolio(day, cash=stash, positions=positions)

        portfolios.append(pf)

        old_positions = positions
        old_stash = stash
        stash, positions = update_positions(prices, day, old_stash,
                                            old_positions, botk, topk)

    return portfolios


def graham_trade(df, prices, clf_name, trading_params):
    k = trading_params['k']

    indices = sorted(
        [day for day in list(set(df.index.values)) if day <= final_date])
    print("Graham trading")

    portfolios = []

    positions = []
    stash = 1000

    for day in indices:
        # we invest in all the ones passing the screening
        topk, botk = graham_screening(df, day=day, k=k)

        pf = Portfolio(day, cash=stash, positions=positions)

        portfolios.append(pf)

        old_positions = positions
        old_stash = stash
        stash, positions = update_positions(prices, day, old_stash,
                                            old_positions, botk, topk)

    return portfolios


@task(returns=dict)
def run_model(clf, clf_name, model_params, df, prices, dataset_name, attrs,
              magic_number, mode, trading_params):
    print("Running model with dataset: %s" % dataset_name)

    df_trade = train(df, attrs, clf, clf_name, model_params, mode,
                     magic_number)

    portfolios = trade(df_trade, prices=prices,
                       clf_name=clf_name, trading_params=trading_params)

    return portfolios


@task(returns=dict)
def run_random(clf, clf_name, model_params, df, prices, dataset_name, attrs,
               magic_number, mode, trading_params):
    print("Running random with dataset: %s" % dataset_name)

    df_trade = train(df, attrs, clf, clf_name, model_params, mode,
                     magic_number)

    print("DF trade:")
    full_print(df_trade)
    portfolios = trade(df_trade, prices=prices,
                       clf_name=clf_name, trading_params=trading_params)

    return portfolios


@task(returns=dict)
def run_baseline(clf, clf_name, model_params, df, prices, dataset_name, attrs,
                 magic_number, mode, trading_params):
    print("Running baseline with dataset: %s" % dataset_name)

    print("Columns: %s" % df.columns)

    res = graham_trade(df=df, prices=prices, clf_name=clf_name,
                       trading_params=trading_params)

    return res


def explore_models(classifiers, df, prices, dataset_name, attrs, save_path,
                   magic_number,
                   trading_params):
    portfolios = defaultdict(list)

    for clf_name, (clf, model_params) in classifiers.items():
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
            run = run_baseline
            mode = REGRESSION

        res = run(clf, clf_name, model_params, df, prices, dataset_name, attrs,
                  magic_number, mode, trading_params)

        portfolios[clf_name].append((model_params, trading_params, res))

    return portfolios
