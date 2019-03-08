from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR

from data_managers.fundamentals_extraction import FundamentalsCollector
from data_managers.price_extraction import PriceExtractor
from data_managers.sic import load_sic
from settings.basic import DATE_FORMAT
from tags import Tags
from utils import load_symbol_list


# while not decided imputation/removals
symbols_list_name = 'sp500'
symbols = load_symbol_list(symbols_list_name)
start_date = '2006-01-01'
end_date = '2019-12-31'
desired_tags = Tags.desired_indicators()
resample_period = '1W'

start_year = int(start_date[0:4])
end_year = int(end_date[0:4])

df_prices = PriceExtractor(symbols_list_name=symbols_list_name,
                           start_date=start_date).collect()

df_fund = FundamentalsCollector(symbols_list_name=symbols_list_name,
                                start_date=start_year,
                                end_date=end_year).collect()

sic_code, sic_industry = load_sic()

# TODO: drop_duplicates an incorrect value from intrinio
df_fund = (df_fund
           .drop_duplicates(['date', 'symbol'], keep='first')
           .assign(date=lambda r: pd.to_datetime(r.date, format=DATE_FORMAT))
           .set_index('date')
           .groupby('symbol')
           .resample(resample_period)
           .ffill()
           .replace('nm', np.NaN)
           .sort_index()
           .assign(
    bookvaluepershare=lambda r: pd.to_numeric(r.bookvaluepershare)))

df_fund = pd.concat(
    [pd.to_numeric(df_fund[col], errors='ignore') for col in df_fund.columns],
    axis=1)
# set common index for outer join

df_prices = (df_prices
             .assign(date=lambda r: pd.to_datetime(r.date, format=DATE_FORMAT))
             .set_index('date')
             .groupby('symbol')
             .resample(resample_period)
             .ffill()
             .sort_index())

available_symbols = set([symbol for symbol, date in df_fund.index.values])
unavailable = [s for s in symbols if s not in available_symbols]
removed_symbols = ['ULTA']
print("Not available symbols: %s\nRemoved symbols: %s" %
      (unavailable, removed_symbols))
for s in removed_symbols:
    available_symbols.remove(s)
merged_dfs = []
for symbol in available_symbols:
    ds = pd.concat([df_fund.loc[symbol], df_prices.loc[symbol]], join='inner',
                   axis=1)

    bins = pd.IntervalIndex.from_tuples(
        [(-np.inf, -0.015), (-0.015, 0.015), (0.015, np.inf)])
    df_tidy = (pd.DataFrame()
               .assign(eps=ds.basiceps,
                       price=ds.price,
                       p2b=ds.price / ds.bookvaluepershare,
                       p2e=ds.price / ds.basiceps,
                       p2r=ds.price / ds.totalrevenue,
                       div2price=pd.to_numeric(
                           ds.cashdividendspershare) / pd.to_numeric(ds.price),
                       divpayoutratio=ds.divpayoutratio,
                       # Performance measures
                       roe=ds.roe,
                       roic=ds.roic,
                       roa=ds.roa,
                       # eva= ???,
                       # Efficiency measures
                       assetturnover=ds.assetturnover,
                       invturnonver=ds.invturnover,
                       profitmargin=ds.profitmargin,
                       debtratio=ds.totalassets / ds.totalliabilities,
                       ebittointerestex=pd.to_numeric(ds.ebit) / pd.to_numeric(
                           ds.totalinterestexpense),
                       # aka times-interest-earned ratio
                       # cashcoverage=ds.ebit + depretitation) / ds.totalinterestexpense,
                       # Liquidity measures
                       wc=ds.nwc,
                       wc2a=pd.to_numeric(ds.nwc) / pd.to_numeric(
                           ds.totalassets),
                       currentratio=ds.totalcurrentassets / ds.totalcurrentliabilities,
                       # Misc. info
                       symbol=symbol,
                       sic_info=sic_code[symbol],
                       sic_industry=sic_industry[symbol],
                       # Target
                       y=(df_prices.loc[symbol].price.shift(1) / ds.price) - 1,
                       positions=lambda r: pd.cut(r.y, bins),
                       next_price=df_prices.loc[symbol].price.shift(1)
                       )
               .set_index('symbol', append=True))

    merged_dfs.append(df_tidy)

del df_fund, df_prices

print("Added weekly indicators.")

# TODO:  there as a paper where they said how to build the survivor bias list
attrs = ['eps', 'p2b', 'p2e', 'p2r', 'div2price',
         'divpayoutratio', 'roe', 'roic', 'roa', 'assetturnover',
         'invturnonver',
         'profitmargin', 'debtratio', 'ebittointerestex', 'wc', 'wc2a',
         'currentratio']

df = (pd.concat(merged_dfs)
      .sort_index())

df.groupby(['date']).size() > 28

print("Adding z-scores...")
# for tag in desired_tags:
dfz = pd.DataFrame(df, copy=True)
dfn = pd.DataFrame(df, copy=True)
for tag in attrs:
    v = df[tag]
    g = df.groupby(['date', 'sic_industry'])[tag]
    dfz[tag] = (v - g.transform(np.mean)) / g.transform(np.std)
    dfn[tag] = v

# columns which have become null, is because they are single groups (can't do z-score) just set a 0 for them
for c in attrs:
    to_fix = dfz[np.isnan(dfz[c]) & ~(np.isnan(dfn[c]))].index.values
    dfz.loc[to_fix, c] = 0

print("Formatting the dataset into x, y for model learning.")
# Drop the NaN
# We drop now, because doing it prior to z-scores loses info.
dfn = (dfn.dropna(axis=0)
       .replace(float('inf'), np.finfo(np.float32).max)
       .replace(-float('inf'), np.finfo(np.float32).min))
dfz = dfz.dropna(axis=0)

# Choose dataset
# df = dfn.reset_index().set_index('date')
df = dfz.reset_index().set_index('date')

idx = 0
indices = sorted(list(set(df.index.values)))

models = defaultdict(list)
classifiers = {'LR': LinearRegression, 'Lasso': Lasso, 'SVM': SVR}

# 53 is a magic number, 52 weeks for training 1 for prediction
while idx + 53 < len(indices) and indices[idx + 53] <= indices[-1]:
    train = df.loc[indices[idx]:indices[idx + 52]]
    test = df.loc[indices[idx + 53]]

    train_x, train_y = train[attrs], train.y
    test_x, test_y = test[attrs], test.y

    # TODO: add DT, RAF, NN, Lasso, LR, SVM,

    print("Training period %s with %s instances." % (idx, train_x.shape[0]))
    for name, model in classifiers.items():
        clf = model().fit(train_x, train_y)

        df.loc[indices[idx + 53], name] = clf.predict(test_x)

        # print("[%s] Score: %s" % (name, clf.score(test_x, test_y)))

        models[name].append(clf.score(test_x, test_y))

    idx += 1

df_trade = df.dropna(axis=0)
money = {tag: [1000] for tag in classifiers.keys()}
choices = {tag: [] for tag in classifiers.keys()}
indices = sorted(list(set(df_trade.index.values)))
k = 10
# topk, botk = k[0], k[0]
for name in classifiers.keys():
    df_clf = df_trade[['y', name]]

    print("Trading for %s" % name)
    for day in indices:
        df_aux = df_clf.loc[day].sort_values(name)

        # TODO add a upper and lower threshold to consider investing
        botk = df_aux.iloc[0:k].query('%s<0' % name)
        topk = df_aux.iloc[-k - 1: -1].query('%s>0' % name)
        # assert botk[botk[name] > 0].shape[0] == 0
        # assert topk[topk[name] < 0].shape[0] == 0

        choices[name].append((topk, botk))
        stash = money[name][-1] / (botk.shape[0] + topk.shape[0])

        long = np.sum(np.add(stash, np.multiply(topk.y, stash)))
        short = np.sum(np.add(stash, np.multiply(botk.y, -stash)))

        money[name].append(long + short)

        print("Day %s: %s (%s, %s)" % (
            day, money[name][-1], botk.shape[0], topk.shape[0]))
        # choices[clf][day] =

results = pd.DataFrame(money, index=[indices[0]] + indices)
