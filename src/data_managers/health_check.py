import pandas as pd

from utils import load_obj


def null_counter(df):
    lst = list(df.isnull().sum())
    return lst.count(0)


def nan_counter(df):
    lst = list(df.isna().sum())
    return lst.count(0)


def nm_counter(df: pd.DataFrame):
    lst = list((df == 'nm').sum())
    return lst.count(0)


def inspect(df):
    nu = df.isnull().sum()
    na = df.isna().sum()
    nm = (df == 'nm').sum()

    nu = pd.DataFrame(nu.reset_index()) \
        .rename(index=str, columns={'index': 'attr', 0: 'isnull()'}) \
        .set_index('attr').sort_index()

    na = pd.DataFrame(na.reset_index()) \
        .rename(index=str, columns={'index': 'attr', 0: 'isna()'}) \
        .set_index('attr').sort_index()

    nm = pd.DataFrame(nm.reset_index()) \
        .rename(index=str, columns={'index': 'attr', 0: 'isnm()'}) \
        .set_index('attr').sort_index()

    result = nu.join(nm)
    # result = result.join(nm)
    result['% isnull()'] = result['isnull()'] / df.shape[0]
    result['% isnm()'] = result['isnm()'] / df.shape[0]
    result['total'] = df.shape[0]

    result = result.round(3)

    return result


def data_health(df: pd.DataFrame):
    print('Non-null cols: %s' % null_counter(df))
    print('Non-nan cols: %s' % nan_counter(df))
    print('Non-nm cols: %s' % nm_counter(df))


def health_check(symbols_list_name='dow30', save=False):
    filename = '../data/csv/%s_monolithic.csv' % symbols_list_name
    all_df = pd.read_csv(filename)

    attrs = load_obj('../data/intrinio_tags')

    groups = attrs.keys()

    # dfs = {}
    # for year in range(2005, 2020):
    #     print("Year: %s" % year)
    #     dfs[year] = all_df.loc[all_df['year'] > year]
    #     data_health(dfs[year])

    for group in groups:
        print("Group %s" % group)
        df_aux = all_df[list(attrs[group])]
        data_health(df_aux)

        res = inspect(df_aux)

        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):
            print(res)

        if save:
            res.to_csv('../data/health_check_%s.csv' % group)


if __name__ == '__main__':
    symbols_list_name = 'sp500'
    res = health_check(symbols_list_name=symbols_list_name)



def check_indices(df):
    print(set(df.index.values))


def stocks_per_index(df):
    indices = sorted(set(df.index.values))
    sizes = [df.loc[idx].shape[0] for idx in indices]
    stats = pd.DataFrame(data=sizes, index=indices, columns=['# stocks'])

    return stats

# THIS WAS ORIGINALLY IN merger_.py BUT WAS DEEMED IRRELEVANT

# THIS ARE ALL HEALTH CHECKS for the data not relevant.
# df_calc = df_fund[[t for t in Tags.calculations if t in df_fund.columns]]
# df_is = df_fund[[t for t in Tags.income_statement if t in df_fund.columns]]
# df_cf = df_fund[[t for t in Tags.cash_flow if t in df_fund.columns]]
# df_bs = df_fund[[t for t in Tags.balance_sheet if t in df_fund.columns]]
#
#
# def missing_info(df, doc):
#     print("\n\n\n[%s] NM per column:" % doc)
#     with pd.option_context('display.max_rows', None, 'display.max_columns',
#                            None):
#         print((df == 'nm').sum())
#     print("\n[%s] NaN per column:" % doc)
#     with pd.option_context('display.max_rows', None, 'display.max_columns',
#                            None):
#         print(df.isna().sum())
#
#
# missing_info(df_calc, 'calculations')
# missing_info(df_is, 'incomse statment')
# missing_info(df_cf, 'cash flows')
# missing_info(df_bs, 'balance sheet')

# after having a look, maybe just dropping the nan/nm is good enough for the documents
# for the calculations, they basically are useless if I need to recompute them
# with the day's share price