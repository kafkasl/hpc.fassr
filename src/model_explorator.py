import os

import numpy as np
import pandas as pd
from sklearn import tree

from data_managers.fundamentals_extraction import FundamentalsCollector
from data_managers.imputation import DataPreprocessor
from data_managers.transformation import IndicatorsBuilder, Indicators




symbols_list_name = 'dow30'
start_year = 2006
end_year = 2019
threshold = 0.015
# experts_criteria = ['graham', 'buffet']
experts_criteria = ['graham', 'all']

dc = FundamentalsCollector(symbols_list_name=symbols_list_name,
                           start_year=start_year,
                           end_year=end_year)


# df['graham'] = DataPreprocessor(dataframe=df_aux).fill_w_value(0).get_df()
# dfs['graham'] = DataPreprocessor(dataframe=df_aux).fill_w_value(0).replace_w_value('nm', float('nan')).drop_any_na().get_df()
# dfs['buffet'] = DataPreprocessor(dataframe=df_aux).fill_w_value(0).replace_w_value('nm', -666).get_df()


indicators = {}
X, y = {}, {}
clfs = {}
graph = {}

axis = {'graham': 0,
        'all': 1}

for expert in experts_criteria:
    df = dc.collect()

    # BUILD GRAHAM DECISION TREE
    # ===============================================================================
    builder = IndicatorsBuilder(df).add_positions(
        threshold=threshold).to_criteria(expert)

    df, target = builder.to_df(), builder.target

    df = DataPreprocessor(dataframe=df).replace_w_value('nm',
                                                        np.NaN).drop_any_na(
        axis[expert]).get_df()

    indicators[expert] = Indicators(df, target)

    X, y = indicators[expert].X, indicators[expert].y

    # X, y = X_df.values, y_df.values

    X_train = X[:-3]
    y_train = y[:-3]
    X_test = X[-3:]
    y_test = y[-3:]

    clf = tree.DecisionTreeClassifier(max_depth=3)

    clf.fit(X, y)

    import graphviz

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=indicators[
                                        expert].feature_names,
                                    class_names=clf.classes_,
                                    filled=True, rounded=True,
                                    special_characters=True)

    graph[expert] = graphviz.Source(dot_data)
    graph[expert].render('%s_%s' % (symbols_list_name, expert), view=True,
                         cleanup=True)

    clfs[expert] = clf
    #
    # os.remove()

    print("\n\n[%s] Features importance:" % expert)
    f_list = ["%s: %.3f" % (f, i) for f, i in list(
        zip(indicators[expert].feature_names, clf.feature_importances_))]

    f_list = sorted(f_list, key=lambda t: t[0])
    # print('\n'.join(["%s: %.3f" % (f, i) for f, i in list(zip(indicators[expert].feature_names, clf.feature_importances_))]))
    print('\n'.join(f_list))
