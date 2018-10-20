import pandas as pd

from utils import to_df, load_obj


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

    nu = pd.DataFrame(nu.reset_index())\
        .rename(index=str, columns={'index': 'attr', 0: 'isnull()'})\
        .set_index('attr').sort_index()

    na = pd.DataFrame(na.reset_index())\
        .rename(index=str, columns={'index': 'attr', 0: 'isna()'})\
        .set_index('attr').sort_index()

    nm = pd.DataFrame(nm.reset_index())\
        .rename(index=str, columns={'index': 'attr', 0: 'isnm()'})\
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
    all_df = to_df(filename)

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

