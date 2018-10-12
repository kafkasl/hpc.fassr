from data_managers.extraction import DataCollector

from data_managers.transformation import IndicatorsBuilder
from utils import to_df

from sklearn import tree
# from models.linear_regression import LinearRegression as model





symbols_list_name = 'debug'
start_year = 2006
end_year = 2019
threshold = 0.015

dc = DataCollector(symbols_list_name=symbols_list_name,
                   start_year=start_year,
                   end_year=end_year)
data = dc.collect()
filename = dc.to_csv()

df = to_df(filename)

# BUILD GRAHAM DECISION TREE
# ===============================================================================
# df = IndicatorsBuilder(df).add_positions(threshold=threshold).to_graham().get_df()
graham_indicators = IndicatorsBuilder(df).add_positions(threshold=threshold).to_graham().build()
# df = graham_indicators.as_dataframe()

# TODO dirty hack, should handle missing values outside here
graham_indicators._df['paymentofdividends'].fillna(0, inplace=True)

X, y = graham_indicators.X, graham_indicators.y

# X, y = X_df.values, y_df.values

X_train = X[:-3]
y_train = y[:-3]
X_test = X[-3:]
y_test = y[-3:]

clf = tree.DecisionTreeClassifier()

clf.fit(X, y)

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=graham_indicators.feature_names,
                         class_names=clf.classes_,
                         filled=True, rounded=True,
                         special_characters=True)

graph = graphviz.Source(dot_data)
graph.render('%s_graham' % symbols_list_name)



# BUILD BUFFET DECISION TREE
# ===============================================================================
# df = IndicatorsBuilder(df).add_positions(threshold=threshold).to_graham().get_df()
buffet_indicators = IndicatorsBuilder(df).add_positions(threshold=threshold).to_buffet().build()
# df = graham_indicators.as_dataframe()

# TODO dirty hack, should handle missing values outside here
# graham_indicators._df['paymentofdividends'].fillna(0, inplace=True)

X, y = buffet_indicators.X, buffet_indicators.y


clf = tree.DecisionTreeClassifier()

clf.fit(X, y)

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=buffet_indicators.feature_names,
                         class_names=clf.classes_,
                         filled=True, rounded=True,
                         special_characters=True)

graph = graphviz.Source(dot_data)
graph.render('%s_buffet' % symbols_list_name)

