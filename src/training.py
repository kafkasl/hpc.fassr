from collections import defaultdict

import numpy as np
import pandas as pd
from pycompss.api.task import task
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR


def trade(df_clf, model, indices, k, top_thresh=0, bot_thresh=0):
    choices, money = [], [1000]
    for day in indices:
        df_aux = df_clf.loc[[day]].sort_values(model)
        # TODO add a upper and lower threshold to consider investing
        botk = df_aux.iloc[0:k].query('%s<%s' % (model, bot_thresh))
        topk = df_aux.iloc[-k - 1: -1].query('%s>%s' % (model, top_thresh))

        choices.append((topk, botk))
        if botk.shape[0] + topk.shape[0] > 0:

            stash = money[-1] / (botk.shape[0] + topk.shape[0])

            long = np.sum(np.add(stash, np.multiply(topk.y, stash)))
            short = np.sum(np.add(stash, np.multiply(botk.y, -stash)))

            money.append(long + short)
        else:
            # no stock worth investing found, keeping my stash
            money.append(money[-1])

        print("Day %s: %s (%s, %s)" % (
            day, money[-1], botk.shape[0], topk.shape[0]))
        # choices[clf][day] =

    return choices, money


@task(returns=object)
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
    indices = sorted(list(set(df_trade.index.values)))
    k = 10
    # topk, botk = k[0], k[0]
    for name in classifiers.keys():
        df_clf = df_trade[['y', name]]

        print("Trading for %s" % name)
        choices[name], money[name] = trade(df_clf, model=name, indices=indices,
                                           k=k)

    results = pd.DataFrame(money, index=[indices[0]] + indices)

    return results
