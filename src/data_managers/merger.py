import numpy as np
import pandas as pd

from data_managers.fundamentals_extraction import FundamentalsCollector
from data_managers.price_extraction import PriceExtractor
from data_managers.sic import load_sic
from settings.basic import DATE_FORMAT
from tags import Tags
from utils import load_symbol_list


def add_new_features_placeholders(df, tags):
    for t in tags:
        df[t] = np.NaN

    return df


def deal_missings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace('nm', np.NaN)
    df = df.dropna(axis=1)
    return df


# while not decided imputation/removals
symbols_list_name = 'dow30'
symbols = load_symbol_list(symbols_list_name)
start_date = '2006-01-01'
end_date = '2019-12-31'
desired_tags = Tags.desired_indicators()

start_year = int(start_date[0:4])
end_year = int(end_date[0:4])

df_prices = PriceExtractor(symbols_list_name=symbols_list_name,
                           start_date=start_date).collect()

df_fund = FundamentalsCollector(symbols_list_name=symbols_list_name,
                                start_year=start_year,
                                end_year=end_year).collect()

sic_info = load_sic(symbols_list_name)

df_fund['date'] = pd.to_datetime(df_fund['date'], format=DATE_FORMAT)
df_prices['date'] = pd.to_datetime(df_prices['date'], format=DATE_FORMAT)

# add sic info
df_fund['sic_code'] = df_fund.apply(lambda r: sic_info[r['symbol']], axis=1)
df_fund['sic_industry'] = df_fund.apply(lambda r: int(r['sic_code'] / 1000),
                                        axis=1)

# set common index for outer join
df_fund = df_fund.set_index(['symbol', 'date'])
df_prices = df_prices.set_index(['symbol', 'date'])

df_fund.sort_index(inplace=True)
df_prices.sort_index(inplace=True)

df_prices = add_new_features_placeholders(df_prices, desired_tags)


def add_weekly_indicators(dfp, row, ps, pe):
    tags2func = {
        Tags.pricetobook: lambda r, row=row:
        r[Tags.price] / row[Tags.bookvaluepershare],

        Tags.pricetorevenue: lambda r, row=row:
        r[Tags.price] / row[Tags.totalrevenue],

        Tags.pricetoearnings: lambda r, row=row:
        r[Tags.price] / row[Tags.basiceps],

        # Tags.dps2price: lambda r:
        # r[Tags.cashdividendspershare] / r[Tags.price],
        #
        # Tags.dividendspayoutratio: lambda r:
        # r[Tags.cashdividendspershare] / r[Tags.basiceps],
        #
        # Tags.roe: lambda r: r[Tags.roe],
        # Tags.roa: lambda r: r[Tags.roa],
        # Tags.profitmargin: lambda r: r[Tags.profitmargin],
        #
        # Tags.debtratio: lambda r:
        # r[Tags.totalliabilities] / r[Tags.totalassets],
        # # Tags.cashcoverageratio: lambda r: (),
        #
        # Tags.currentratio: lambda r:
        # r[Tags.totalcurrentassets] / r[Tags.totalcurrentliabilities]
    }
    for tag, func in tags2func.items():
        dfp.loc[ps:pe, tag] = dfp[ps:pe].apply(func, axis=1)


new_dfs = []
symbols.remove('DWDP')
for symbol in symbols:
    # try:
    dff = df_fund.loc[symbol]
    dfp = df_prices.loc[symbol]

    dfp[Tags.next_price] = dfp.[Tags.price].shift(1)

    for (date, row) in dff.iterrows():
        ps, pe = row[Tags.rep_period].split(":")

        add_weekly_indicators(dfp, row, ps, pe)

    # dfp = dfp.dropna(axis=0)

    if dfp.shape[0] > 0:
        dfp[Tags.symbol] = symbol

        # add sic info
        dfp['sic_code'] = dfp.apply(lambda r: sic_info[r['symbol']], axis=1)
        dfp['sic_industry'] = dfp.apply(lambda r: int(r['sic_code'] / 1000),
                                        axis=1)

        new_dfs.append(dfp)
        # except ValueError as e:
        #     print("Error processing %s: %s" % (symbol, e))

print("Added weekly indicators. Proceeding to merge them into a single df.")
df = pd.concat(new_dfs)
df = df.reset_index()
df['date'] = pd.to_datetime(df['date'], format=DATE_FORMAT)
df = df.set_index(['date', 'symbol'])
df = df.sort_index()


def z_score(df, date, tag, sic):
    res = df.loc[date].apply(lambda r: (r[tag]
                                        - np.mean(
        df.loc[date].query('sic_industry==%s' % r[sic])[tag]))
                                       / np.std(
        df.loc[date].query('sic_industry==%s' % r[sic])[tag]), axis=1)

    return res


print('Adding z-scores.')

sic = 'sic_industry'

dates = list(set([d.strftime(DATE_FORMAT) for d, s in df.index.values]))
for date in dates:
    for tag in desired_tags:
        df.loc[date, 'z-' + tag] = z_score(df, date, tag, sic)



print("Formatting the dataset into x, y for model learning.")
# Drop the NaN
x_indicators = [Tags.pricetobook, Tags.pricetorevenue, Tags.pricetoearnings]
df_filtered = df[x_indicators + [Tags.next_price]].dropna(axis=0)
target = Tags.next_price
x, y = df_filtered[x_indicators].values, df_filtered[target].values



# sics = set(df[z_tags])
# dates = set([d for d, s in df.index.values])
# for sic in sics:
#     for date in dates:
#         mean = np.mean()



# for ((date, symbol), row) in df.iterrows():
# day_df = df.loc[date]
# df_aux = day_df[day_df['sic_industry'] == row['sic_industry']]
# mean = np.
# df.loc[date].query('sic_industry==%s' % s)
#
# mean, std = {}, {}
#
# for tag in [Tags.pricetobook, Tags.pricetoearnings, Tags.pricetorevenue]:
#     for
# sic_industry in set(final_df['sic_industry'].values):
# values = final_df.loc[final_df['sic_industry'] == sic_industry]
# mean[tag + '-' + sic_industry] = np.mean(values)
# std[tag + '-' + sic_industry] = np.std(values)
# mean[tag + '-' + sic_industry] = np.mean(values)
# std[tag + '-' + sic_industry] = np.std(values)

# final_df['z-%s' % Tags.pricetobook] = final_df.apply(
#     lambda r: (r[tag] - np.mean(values)) / np.std(values), axis=1)
