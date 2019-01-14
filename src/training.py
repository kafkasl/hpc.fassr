from collections import defaultdict

import numpy as np
import pandas as pd
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR

from settings.basic import DATE_FORMAT


def trade(df_trade, clf_name, k, top_thresh=0, bot_thresh=0):
    indices = sorted(list(set(df_trade.index.values)))

    df_clf = df_trade[['y', clf_name, 'symbol']]

    print("Trading for %s" % clf_name)
    choices, money = [], [1000]
    for day in indices:
        df_aux = df_clf.loc[[day]].sort_values(clf_name)
        # TODO add a upper and lower threshold to consider investing
        botk = df_aux.iloc[0:k].query('%s<%s' % (clf_name, bot_thresh))
        topk = df_aux.iloc[-k - 1: -1].query('%s>%s' % (clf_name, top_thresh))

        str_date = pd.to_datetime(str(day)).strftime(DATE_FORMAT)
        choices.append(
            (str_date, list(topk.symbol.values), list(botk.symbol.values)))
        if botk.shape[0] + topk.shape[0] > 0:

            stash = money[-1] / (botk.shape[0] + topk.shape[0])

            long = np.sum(np.add(stash, np.multiply(topk.y, stash)))
            short = np.sum(np.add(stash, np.multiply(botk.y, -stash)))

            money.append(long + short)
        else:
            # no stock worth investing found, keeping my stash
            money.append(money[-1])

            # print("Day %s: %s (%s, %s)" % (
            #     day, money[-1], botk.shape[0], topk.shape[0]))
            # choices[clf][day] =

    return choices, money


@task(returns=2)
def run_model(df, name, attrs, magic_number=53):
    print("Trading with dataset: %s" % name)
    idx = 0
    indices = sorted(list(set(df.index.values)))

    models = defaultdict(list)
    classifiers = {'LR': LinearRegression, 'Lasso': Lasso, 'SVM': SVR}

    # magic number is by default 53, 52 weeks for training 1 for prediction
    while idx + magic_number < len(indices) and indices[idx + magic_number] <= \
            indices[-1]:
        train = df.loc[indices[idx]:indices[idx + magic_number - 1]]

        test_idx = indices[idx + magic_number]
        if type(test_idx) is not list:
            test_idx = [test_idx]
        test = df.loc[test_idx]

        train_x, train_y = train[attrs], train.y
        test_x, test_y = test[attrs], test.y

        # TODO: add DT, RAF, NN, Lasso, LR, SVM,

        print(
            "Training period %s with %s instances." % (idx, train_x.shape[0]))
        for name, model in classifiers.items():
            clf = model().fit(train_x, train_y)

            # print("TEST_X")
            # print(test_x.shape)
            # print("len: %s" % len(test_x.shape))
            # print(test_x)

            df.loc[indices[idx + magic_number], name] = clf.predict(test_x)

            # print("[%s] Score: %s" % (name, clf.score(test_x, test_y)))

            models[name].append(clf.score(test_x, test_y))

        idx += 1

    df_trade = df.dropna(axis=0)

    money = {}
    choices = {}
    # money = {tag: [1000] for tag in classifiers.keys()}
    # choices = {tag: [] for tag in classifiers.keys()}
    k = 10
    # topk, botk = k[0], k[0]
    for name in classifiers.keys():
        choices[name], money[name] = trade(df_trade, name, k)

    indices = sorted(list(set(df_trade.index.values)))

    res_choices, money = compss_wait_on(choices), compss_wait_on(money)

    res_money = pd.DataFrame(money, index=[indices[0]] + indices)

    return res_money, res_choices
