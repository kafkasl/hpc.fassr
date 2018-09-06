import pickle
import itertools
import csv

import numpy as np
import pandas as pd

from settings.basic import DATE_FORMAT
from utils.cache import call_and_cache
from string import Template
from collections import OrderedDict
from collections import defaultdict
from dateutil.relativedelta import relativedelta
from datetime import datetime
from typing import Callable

"""
The goal of this file is to download a fundamental dataset with the following features:
* Contains all possible reportable attributes of income statement, cash flow, balance sheet.
* N/A will be coded as -1 (after checking that no -1 value exists).
* Check if calculations / current offer also interesting values.

"""


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_start_date(end_date_str: str, delta: relativedelta) -> str:
    end_date = datetime.strptime(end_date_str, DATE_FORMAT)

    start_date = end_date - delta

    return start_date.strftime(DATE_FORMAT)


fundamentals_url = Template("https://api.intrinio.com/financials/standardized?"
                            "identifier=${symbol}&"
                            "statement=${statement}&"
                            "type=TTM&"
                            "fiscal_year=${year}"
                            "period=${period}")

symbols_list = 'dow30'
# symbols_list = 'sp500'
start_year = 2006
end_year = 2019
year_range = list(range(start_year, end_year))
periods = ["Q1TTM", "Q2TTM", "Q3TTM", "FY"]
statements = ["income_statement", "balance_sheet", "cash_flow_statement"]

quarters = [1, 2, 3, 4]
quarter2index = {'Q1TTM': 1, 'Q2TTM': 2, 'Q3TTM': 3, 'FY': 4}

delta_months = 1
delta = relativedelta(months=delta_months)

period_keys = ['%s-%s' % (y, q) for y, q in itertools.product(range(2006, 2020), quarters)]

period2index = {period: index for index, period in enumerate(period_keys)}
index2period = {index: period for index, period in enumerate(period_keys)}

# TODO: remember to update dow30 with ALL symbols ever to be there to avoid survivor bias

symbols = open('../data/%s_symbols.lst' % symbols_list).read().split()
# symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT']

series_financials_dict = {}
series_periods = set()
for symbol in symbols:
    period_dict = {}
    for year in year_range:
        for period in periods:

            symbol_dict = {}

            for statement in statements:
                url = fundamentals_url.substitute(symbol=symbol, statement=statement, year=year,
                                                  period=period)

                data_json = call_and_cache(url)

                statement_dict = {}
                elements = []
                while True:
                    try:
                        elements = data_json['data']
                        break
                    except KeyError as e:
                        print("LCS: %s" % e)
                        pass
                if len(elements) == 0:
                    print("LCS: No data in %s" % url)
                for element in elements:
                    statement_dict[element['tag']] = element['value']

                symbol_dict.update(statement_dict)

            period_key = "%s-%s" % (year, quarter2index[period])
            period_dict[period2index[period_key]] = symbol_dict

        # add the data of all symbols for this period (e.g. 2017-Q1)
        series_financials_dict[symbol] = period_dict

save_obj(series_financials_dict, '../data/%s_%s-%s_financials' % (symbols_list, start_year,
                                                                  end_year))

# ============================================================================================
# Compute the average 3-months price
stock_price_url = Template("https://api.intrinio.com/prices?"
                           "identifier=${symbol}&"
                           "start_date=${start_date}&"
                           "end_date=${end_date}&"
                           "frequency=daily")

report_periods_url = Template("https://api.intrinio.com/fundamentals/standardized?"
                              "identifier=${symbol}&"
                              "statement=income_statement&"
                              "type=TTM")

series_price_dict = defaultdict(dict)

for symbol in symbols:
    url = report_periods_url.substitute(symbol=symbol)

    data_json = call_and_cache(url)

    for period_info in data_json['data']:

        end_date = period_info['end_date']

        start_date = get_start_date(end_date, delta)

        year = period_info['fiscal_year']
        fiscal_period = period_info['fiscal_period']
        period_key = "%s-%s" % (year, quarter2index[fiscal_period])

        # Check that financial data is available (otherwise skip getting stock's price)
        if period2index[period_key] in series_financials_dict[symbol]:
            url = stock_price_url.substitute(symbol=symbol, start_date=start_date,
                                             end_date=end_date)
            data_json = call_and_cache(url)

            quarter_mean = np.mean([day_price['close'] for day_price in data_json['data']])

            series_price_dict[symbol][period2index[period_key]] = quarter_mean

save_obj(series_price_dict, '../data/%s_%s-%s_T%sM_price' % (symbols_list, start_year, end_year,
                                                             delta_months))

# Create dataset

# Find all possible attributes reported by either of the 3 documents
all_attrs = set()
for symbol in series_financials_dict.keys():
    for period in series_financials_dict[symbol].keys():
        all_attrs = all_attrs.union(list(series_financials_dict[symbol][period].keys()))

# Create a mapping to transform the dicts into matrix the 0 is the stock name
attr2id = {'stock': 0, '%sM increase' % delta_months: len(all_attrs)}
attr2id.update({attr: i + 1 for i, attr in enumerate(all_attrs)})

# Write CSVs split by period
for period_idx in period2index.values():
    period_name = index2period[period_idx]

    file_name = '../data/%s_%s.csv' % (symbols_list, period_name)
    # Check that are stocks fulfilling the conditions (i.e. having fundamentals and future
    # price)
    available = [period_idx in series_financials_dict[symbol].keys() and
                 period_idx + 1 in series_price_dict[symbol].keys()
                 for symbol in symbols].count(True)

    if available > 0:
        with open(file_name, mode='w') as csv_file:
            fieldnames = attr2id.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()

            for symbol in symbols:
                if (period_idx in series_financials_dict[symbol].keys()) and \
                        (period_idx + 1 in series_price_dict[symbol].keys()) and \
                        (period_idx in series_price_dict[symbol].keys()):
                    incr = series_price_dict[symbol][period_idx + 1] / series_price_dict[symbol][
                        period_idx]
                    row = {'stock': symbol, '%sM increase' % delta_months: incr}
                    row.update(series_financials_dict[symbol][period_idx])
                    writer.writerow(row)

# Write monolithic CSV
file_name = '../data/%s_monolithic.csv' % symbols_list
attr2id.update({'year': len(attr2id) - 1, 'quarter': len(attr2id)})

with open(file_name, mode='w') as csv_file:
    fieldnames = attr2id.keys()
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    for period_idx in period2index.values():
        period_name = index2period[period_idx]

        year, quarter = period_name.split('-')

        for symbol in symbols:
            if (period_idx in series_financials_dict[symbol].keys()) and \
                    (period_idx + 1 in series_price_dict[symbol].keys()) and \
                    (period_idx in series_price_dict[symbol].keys()):
                incr = (series_price_dict[symbol][period_idx + 1] / series_price_dict[symbol][
                    period_idx]) - 1

                row = {'stock': symbol, 'year': year, 'quarter': quarter,
                       '%sM increase' % delta_months: incr}
                row.update(series_financials_dict[symbol][period_idx])
                writer.writerow(row)


                # def transform_to_numpy(data_csv: pd.DataFrame,
                #                        imputation_func: Callable[[pd.DataFrame], pd.DataFrame]) -> \
                # def transform_to_numpy(df: pd.DataFrame) -> np.ndarray:
                # Pandas


def to_df(file: str) -> pd.DataFrame:
    df = pd.read_csv(file)

    # df.set_index(['stock', 'year', 'quarter'], inplace=True)
    df.set_index(['year', 'quarter'], inplace=True)
    df.sort_index(inplace=True)

    # Convert to numpy

    # for c in data_imp.columns:
    #     print("%s" % (len(data_imp[c]) - len(data_imp[c].isnull().count(True))))

    # Need to decide an imputation
    # data = imputation_func(data_csv)  # for all dataset
    # data[c].fillna('')  # for a single 'c' column

    return df


df = to_df('../data/%s_monolithic.csv' % symbols_list)

# Imputation methods
fill_w_0 = lambda df: df.fillna(0)
# drop_all = lambda df: df.dropna(how='all')
drop_any_na = lambda df: df.dropna(how='any')


def drop_some(df_: pd.DataFrame, thresh: int) -> pd.DataFrame:
    # thresh is the minimum number of NA, the 1 indicates that columns should be dropped not rows
    return df_.dropna(1, thresh=thresh)

# TODO remember to scale

df = drop_some(df, int(len(df) * 0.85))
df = fill_w_0(df)



# Model training


import tensorflow as tf
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
from tensorflow.python.saved_model import tag_constants

# remove stock name, year and quarter for now
df_v = df.drop(['stock'], axis=1)
X = scale(df_v.loc[:, df_v.columns != '1M increase'].values)
Y = df_v['1M increase'].values

train_ratio = 0.7
valid_ratio = 0.1
test_ratio = 0.2

train_idx = int(train_ratio * len(X))
valid_idx = int((train_ratio + valid_ratio) * len(X))

train_X = X[0: train_idx]
train_Y = Y[0: train_idx]

valid_X = X[train_idx: valid_idx]
valid_Y = Y[train_idx: valid_idx]

test_X = X[valid_idx:]
test_Y = Y[valid_idx:]


# Parameters
learning_rate = 0.01
training_epochs = 10000
display_step = 50

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder(tf.float64, shape=(None, train_X.shape[1]))
Y = tf.placeholder(tf.float64)

# Set model weightsf

W = tf.Variable(tf.truncated_normal([train_X.shape[1], 1], mean=0.0, stddev=1.0, dtype=tf.float64))
b = tf.Variable(tf.zeros(1, dtype=tf.float64))
# Construct a linear model
pred = tf.add(tf.matmul(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.square(pred - Y)) / (2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x.reshape(1, train_X.shape[1]), Y: y})

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W=", sess.run(W),
                  "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')


    tf.saved
    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(tf.add(tf.matmul(train_X, sess.run(W)), sess.run(b))),
                                      label='Fitted line')
    plt.legend()
    plt.show()

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
