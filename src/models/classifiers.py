from itertools import product

# from autosklearn.classification import AutoSklearnClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# TODO: check if adding 'revenue', 'epsgrowth', 'bvps' helps or not.
train_attrs = ['eps', 'p2b', 'p2e', 'p2r', 'div2price',
               'divpayoutratio', 'roe', 'roic', 'roa', 'assetturnover',
               'invturnonver', 'profitmargin', 'debtratio', 'ebittointerestex',
               'wc', 'wc2a', 'currentratio', 'revenue', 'epsgrowth', 'bvps']

reg_classifiers = {'LR': (LinearRegression, [{}]),
                   'MLPR': (MLPRegressor, [{}]),
                   'SVR': (SVR, [{}]),
                   'AdaBR': (AdaBoostRegressor, [{}]),
                   'DTR': (DecisionTreeRegressor, [{}]),
                   'RFR': (RandomForestRegressor, [{}])}
cat_classifiers = {'MLPC': (MLPClassifier, [{}]),
                   'SVC': (SVC, [{}]),
                   'AdaBC': (AdaBoostClassifier, [{}]),
                   'GBC': (GradientBoostingClassifier, [{}]),
                   'DTC': (DecisionTreeClassifier, [{}]),
                   'RFC': (RandomForestClassifier, [{}]),
                   'ExtraTC': (ExtraTreesClassifier, [{}])}
                   # 'AutoC': (AutoSklearnClassifier, [{}])}

random = {'random': (LinearRegression, [{}])}
debug = {'debug': (LinearRegression, [{}])}
graham = {'graham': (LinearRegression, [{}])}

baseline_classifiers = {**random, **graham}

debug_classifiers = {**baseline_classifiers, **{'SVC': (SVC, [{}])}}

all_classifiers = {**reg_classifiers, **cat_classifiers,
                   **baseline_classifiers}

# C = [{'C': 2 ** i} for i in [0.03125, 0.125]]
C = [{'C': 2 ** i} for i in range(-5, 15, 2)]
# C.extend([{'C': 1.0}])

gamma = [{'gamma': 2 ** i} for i in range(-15, 3, 2)]
gamma.extend([{'gamma': 'auto'}])
# gamma = [{'gamma': 'auto'}]
svc_params = [{**d1, **d2} for d1, d2 in product(C, gamma)]

# n_samples = [90, 700, 1000, 1300]
# n_input = [len(train_attrs)]
# n_output = [1, 3]
# alpha = [2, 5, 7, 10]
# n_h = list(set([int(ns / (alp * (ni + no)))
#                 for ns, ni, no, alp in
#                 product(n_samples, n_input, n_output, alpha)]))\
n_h = [{'hidden_layer_sizes': [n]} for n in
       [5, 10, 15, 20, 30, 50, 100, 200, 500]]
m_solver = [{'solver': s} for s in ['lbfgs', 'adam']]
mlp_params = [{**d1, **d2} for d1, d2 in product(n_h, m_solver)]

# using 250, 500 or 1000 yielded similar results
raf_params = [{'n_estimators': i, 'n_jobs': 8} for i in [50, 100, 250, 500]]
ada_params = [{'n_estimators': i} for i in [50, 100]]
gbc_params = [{'n_estimators': i} for i in [50, 100]]

# debug_1_classifiers = {'AutoC': (AutoSklearnClassifier, [{}]),
# debug_1_classifiers = {'RFC': (RandomForestClassifier, raf_params),
#                        'ExtraTC': (ExtraTreesClassifier, raf_params),
#                        'GBC': (GradientBoostingClassifier, gbc_params)}

debug_1_classifiers = {**debug, **{'SVC': (SVC, [{'C': 0.125, 'gamma': 0.125}])}}
# debug_1_classifiers = {**debug, **{'SVC': (SVC, [{'C': 0.125, 'gamma': 0.125}])}}
                       # 'RFC': (RandomForestClassifier, [{'n_estimators': 100, 'n_jobs': 2}])}
# debug_1_classifiers = {'GBC': (GradientBoostingClassifier, gbc_params)}
# debug_1_classifiers = {'AdaBC': (AdaBoostClassifier, [{'n_estimators': 1000}])}
# debug_1_classifiers = {**graham,
#                        **{'SVC': (SVC, [{}]),
#                           'MLPC': (MLPClassifier, [{}]),
#                           'RFC': (RandomForestClassifier, [{}]),
#                           'AdaBC': (AdaBoostClassifier, [{'n_estimators': 1000}])}}

experiment_1_classifiers = {**graham,
                            **{'SVC': (SVC, svc_params),
                               'MLPC': (MLPClassifier, mlp_params),
                               'RFC': (RandomForestClassifier, raf_params),
                               'AdaBC': (AdaBoostClassifier, ada_params)}}

experiment_2_classifiers = {**graham,
                            **{'SVC': (SVC, svc_params),
                               'MLPC': (MLPClassifier, mlp_params),
                               'RFC': (RandomForestClassifier, raf_params),
                               'AdaBC': (AdaBoostClassifier, ada_params),
                               'LR': (LinearRegression, [{}]),
                               'MLPR': (MLPRegressor, mlp_params),
                               'SVR': (SVR, svc_params),
                               'AdaBR': (AdaBoostRegressor, ada_params),
                               'RFR': (RandomForestRegressor, raf_params)
                               }}
