import os
import sys

import numpy as np
import pandas as pd
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task

from data_managers.fundamentals_extraction import FundamentalsCollector
from data_managers.price_extraction import PriceExtractor
from data_managers.sic import load_sic
from models.classifiers import train_attrs as attrs
from settings.basic import DATE_FORMAT, DATA_PATH
from utils import load_symbol_list, save_obj, exists_obj, get_datasets_name

try:
    import pyextrae.multiprocessing as pyextrae

    tracing = True
except:
    tracing = False


@task(returns=pd.DataFrame)
def get_prices(symbols_list_name, start_date='2006-01-01',
               resample_period='1W', only_prices=False):
    if tracing:
        pro_f = sys.getprofile()
        sys.setprofile(None)

    prices = _get_prices(symbols_list_name, start_date, resample_period)

    if only_prices:
        res = prices.price
    else:
        res = prices

    if tracing:
        sys.setprofile(pro_f)

    return res


def _get_prices(symbols_list_name, start_date='2006-01-01',
                resample_period='1W'):
    print("Loading prices for %s [%s - end] %s" % (
        symbols_list_name, start_date, resample_period))
    df_prices = PriceExtractor(symbols_list_name=symbols_list_name,
                               start_date=start_date).collect()

    # set common index for outer join

    df_prices = (df_prices
                 .assign(
        date=lambda r: pd.to_datetime(r.date, format=DATE_FORMAT))
                 .set_index('date')
                 .groupby('symbol')
                 .resample(resample_period)
                 .ffill()
                 .sort_index())

    return df_prices


@task(returns=pd.DataFrame)
def get_fundamentals(symbols_list_name, start_date, end_date, resample_period):
    if tracing:
        pro_f = sys.getprofile()
        sys.setprofile(None)

    print("Loading fundamentals for %s [%s - %s] %s" % (
        symbols_list_name, start_date, end_date, resample_period))
    df_fund = FundamentalsCollector(symbols_list_name=symbols_list_name,
                                    start_date=start_date,
                                    end_date=end_date).collect()
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
        [pd.to_numeric(df_fund[col], errors='ignore') for col in
         df_fund.columns],
        axis=1)

    if tracing:
        sys.setprofile(pro_f)

    return df_fund


# @task(returns=pd.DataFrame)
def process_symbol(symbol, df_fund, df_prices, sic_code, sic_industry,
                   thresholds, target_shift):
    # TODO remove this once pyCOMPSs supports single-char parameters
    symbol = symbol[:-1]
    bot_thresh, top_thresh = thresholds
    print("Processing symbol [%s]" % symbol)
    ds = pd.concat([df_fund.loc[symbol], df_prices.loc[symbol]],
                   join='inner',
                   axis=1)

    bins = pd.IntervalIndex.from_tuples(
        [(-np.inf, bot_thresh), (bot_thresh, top_thresh),
         (top_thresh, np.inf)])

    df_tidy = (pd.DataFrame()
               .assign(eps=ds.basiceps,
                       price=ds.price,
                       p2b=ds.price / ds.bookvaluepershare,
                       p2e=ds.price / ds.basiceps,
                       p2r=ds.price / ds.totalrevenue,
                       div2price=pd.to_numeric(
                           ds.cashdividendspershare) / pd.to_numeric(
                           ds.price),
                       divpayoutratio=ds.divpayoutratio,
                       # Performance measures
                       roe=ds.roe,
                       roic=ds.roic,
                       roa=ds.roa,
                       # Efficiency measures
                       assetturnover=ds.assetturnover,
                       invturnonver=ds.invturnover,
                       profitmargin=ds.profitmargin,
                       debtratio=ds.totalassets / ds.totalliabilities,
                       ebittointerestex=pd.to_numeric(
                           ds.ebit) / pd.to_numeric(
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
                       # Graham screening
                       revenue=ds.operatingrevenue,
                       epsgrowth=ds.epsgrowth,
                       bvps=ds.bookvaluepershare,
                       # Target
                       y=(df_prices.loc[symbol].price.shift(
                           target_shift) / ds.price) - 1,
                       positions=lambda r: pd.cut(r.y, bins).cat.codes - 1,
                       )
               .set_index('symbol', append=True))

    return df_tidy





@task(returns=1)
def process_symbols(available_symbols, df_fund, df_prices, sic_code,
                    sic_industry, thresholds, target_shift):
    if tracing:
        pro_f = sys.getprofile()
        sys.setprofile(None)

    merged_dfs = []

    for i, symbol in enumerate(available_symbols):
        merged_dfs.append(process_symbol(symbol=symbol + '_', df_fund=df_fund,
                                         df_prices=df_prices,
                                         sic_code=sic_code,
                                         sic_industry=sic_industry,
                                         thresholds=thresholds,
                                         target_shift=target_shift))

    df = pd.concat(merged_dfs).sort_index()

    if tracing:
        sys.setprofile(pro_f)

    return df


@task(returns=2)
def post_process(df, files):
    # TODO:  there is a paper where they said how to build the survivor bias list
    if tracing:
        pro_f = sys.getprofile()
        sys.setprofile(None)

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
           .replace(float('inf'), np.finfo(np.float16).max)
           .replace(-float('inf'), np.finfo(np.float16).min)
           .reset_index().set_index('date'))
    dfz = dfz.dropna(axis=0).reset_index().set_index('date')

    if not exists_obj(files[0]):
        save_obj(dfn, files[0])
    if not exists_obj(files[1]):
        save_obj(dfz, files[1])

    if tracing:
        sys.setprofile(pro_f)

    return dfn, dfz

def get_data(thresholds, resample_period='1W', symbols_list_name='sp500',
             start_date='2006-01-01', target_shift=4):
    print("Getting data for: %s - %s from %s with thresholds %s" % (
        symbols_list_name, resample_period, start_date, list(thresholds)))

    # while not decided imputation/removals
    symbols = load_symbol_list(symbols_list_name)
    end_date = '2019-12-31'

    df_prices = get_prices(symbols_list_name=symbols_list_name,
                           start_date=start_date,
                           resample_period=resample_period)

    df_fund = get_fundamentals(symbols_list_name=symbols_list_name,
                               start_date=start_date,
                               end_date=end_date,
                               resample_period=resample_period)

    sic_code, sic_industry = load_sic(symbols_list_name=symbols_list_name)

    alist_path = os.path.join(DATA_PATH, 'available_%s' % symbols_list_name)

    if os.path.isfile(alist_path):
        available_symbols = [l.strip() for l in open(alist_path).readlines()]
    else:
        df_fund = compss_wait_on(df_fund)
        available_symbols = set(
            [symbol for symbol, date in df_fund.index.values])
        unavailable = [s for s in symbols if s not in available_symbols]
        removed_symbols = ['ULTA']
        print("Not available symbols: %s\nRemoved symbols: %s" %
              (unavailable, removed_symbols))

        for s in removed_symbols:
            try:
                available_symbols.remove(s)
            except KeyError:
                print("Couldn't remove symbol %s" % s)

        with open(os.path.join(DATA_PATH, 'available_%s' % symbols_list_name),
                  'w') as f:
            f.write('\n'.join(available_symbols))

    df = process_symbols(available_symbols, df_fund, df_prices, sic_code,
                         sic_industry, thresholds, target_shift)

    normal_name, z_name = get_datasets_name(resample_period, symbols_list_name,
                                            thresholds, target_shift)

    normal_file = os.path.join(DATA_PATH, normal_name)
    z_file = os.path.join(DATA_PATH, z_name)

    res = post_process(df, (normal_file, z_file))

    return res
