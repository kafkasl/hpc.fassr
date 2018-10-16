import os

import graphviz
import numpy as np
from sklearn import tree

from data_managers.extraction import DataCollector
from data_managers.imputation import DataPreprocessor
from data_managers.transformation import IndicatorsBuilder, Indicators
from utils import to_df


def export_tree(clf, feature_names, name):
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=clf.classes_,
                                    filled=True, rounded=True,
                                    special_characters=True)

    graph = graphviz.Source(dot_data)
    graph.render(name, cleanup=True)


def imputate(df):
    nc = df.columns[df.isna().any()].tolist()
    if len(nc) > 0:
        print("MISSING DATA: Columns with null values %s" % df.columns[
            df.isna().any()].tolist())

    df = DataPreprocessor(dataframe=df) \
        .replace_w_value('nm', np.NaN) \
        .drop_any_na(axis=0).get_df()

    return df


symbols_list_name = 'dow30'
start_year = 2006
end_year = 2019
threshold = 0.015
# experts_criteria = ['graham', 'buffet']
expert = 'graham'
imputation_active = False

dc = DataCollector(symbols_list_name=symbols_list_name,
                   start_year=start_year,
                   end_year=end_year)

filename = dc.csv_filename()

if not os.path.isfile(filename):
    data = dc.collect()
    filename = dc.to_csv()

all_df = to_df(filename)
symbols = set(all_df['symbol'])

print("Symbols to be processed: %s" % symbols)
symbol2clf = {}

for symbol in symbols:
    try:
        print("\n\nProcessing symbol: %s\n" % symbol)
        symbol_df = all_df.loc[all_df['symbol'] == symbol]
        symbol_df.set_index(['current_date'])

        print("Number of rows: %s\n" % (symbol_df.shape[0]))
        builder = IndicatorsBuilder(symbol_df) \
            .add_positions(threshold=threshold) \
            .to_criteria(expert)

        df, target = builder.to_df(), builder.target

        if imputation_active:
            df = imputate(df)
        i = Indicators(df, target)

        X, y = i.X, i.y

        clf = tree.DecisionTreeClassifier()

        clf.fit(X, y)
        symbol2clf[symbol] = clf

        graph_name = '%s_%s_%s' % (symbol, symbols_list_name, expert)
        export_tree(clf, i.feature_names, graph_name)

        print("Features importance:")
        f_list = ["%s: %.3f" % (f, i) for f, i in list(
            zip(i.feature_names, clf.feature_importances_))]
        f_list = sorted(f_list, key=lambda t: t[0])
        print('\n'.join(f_list))

    except ValueError as e:
        print("Error processing %s.\n%s" % (symbol, e))
