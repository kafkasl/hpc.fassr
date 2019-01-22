import os
from collections import defaultdict

import matplotlib
from pycompss.api.task import task
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from models.portfolio import Portfolio, Position
from settings.basic import logging

matplotlib.use('Agg')

REGRESSION = 'REG'
CLASSIFICATION = 'CAT'


def save(items, filename, path):
    file_path = os.path.join(path, filename)
    logging.debug("Saving results %s to %s" % (filename, file_path))

    items.to_csv("%s.csv" % file_path)
    plot = items.plot(title=filename)
    fig = plot.get_figure()
    fig.savefig("%s.png" % file_path)


def get_share_price(prices, day, symbol):
    return prices.loc[(symbol, day)]


def get_k_best(df_aux, clf_name, top_thresh, bot_thresh, k):
    botk = df_aux.iloc[0:k].query('%s<%s' % (clf_name, bot_thresh))
    topk = df_aux.iloc[-k - 1: -1].query('%s>%s' % (clf_name, top_thresh))

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
        # intially divide money equally among candidates
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


def trade(df_trade, prices, clf_name, k, top_thresh, bot_thresh):
    indices = sorted(list(set(df_trade.index.values)))
    df_clf = df_trade[['y', clf_name, 'symbol']]
    logging.debug("Trading for %s" % clf_name)

    portfolios = []

    positions = []
    stash = 1000

    for day in indices:
        # sort by predicted increment per stock of classifier clf_name
        df_aux = df_clf.loc[[day]].sort_values(clf_name)

        topk, botk = get_k_best(df_aux, clf_name, top_thresh, bot_thresh, k)

        pf = Portfolio(day, cash=stash, positions=positions)

        portfolios.append(pf)

        old_positions = positions
        old_stash = stash
        stash, positions = update_positions(prices, day, old_stash,
                                            old_positions,
                                            botk, topk)

    return portfolios


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
        logging.debug(
            "Training %s with %s instances." % (idx, train_x.shape[0]))

        clf_ = clf(**models_params).fit(train_x, train_y)

        df.loc[indices[idx + magic_number], clf_name] = clf_.predict(test_x)

        idx += 1

    df_trade = df.dropna(axis=0)

    return df_trade


@task(returns=dict)
def run_model(clf, clf_name, model_params, df, prices, dataset_name, attrs,
              magic_number, mode, k, top_thresh=0,
              bot_thresh=0):
    print("Trading with dataset: %s" % dataset_name)

    df_trade = train(df, attrs, clf, clf_name, model_params, mode,
                     magic_number)

    portfolios = trade(df_trade, prices=prices,
                       clf_name=clf_name, k=k,
                       top_thresh=top_thresh,
                       bot_thresh=bot_thresh)

    return portfolios


def explore_models(df, prices, dataset_name, attrs, save_path, magic_number=53,
                   k=10):
    reg_classifiers = {'LR': LinearRegression, 'Lasso': Lasso, 'SVR': SVR,
                       'AdaBR': AdaBoostRegressor,
                       'DTR': DecisionTreeRegressor,
                       'RFR': RandomForestRegressor}
    cat_classifiers = {'NN': MLPClassifier, 'SVM': SVC,
                       'AdaBC': AdaBoostClassifier,
                       'DTC': DecisionTreeClassifier,
                       'RFC': RandomForestClassifier}

    portfolios = defaultdict(list)
    model_params = {}

    for clf_name, clf in reg_classifiers.items():
        res = run_model(clf, clf_name, model_params, df, prices, dataset_name,
                        attrs, magic_number, REGRESSION, k)

        portfolios[clf_name].append((model_params, res))

    for clf_name, clf in cat_classifiers.items():
        res = run_model(clf, clf_name, model_params, df, prices, dataset_name,
                        attrs, magic_number, CLASSIFICATION, k)

        portfolios[clf_name].append((model_params, res))

    return portfolios
