from collections import OrderedDict
from itertools import product

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

C = [{'C': 2 ** i} for i in range(-5, 15, 2)]

gamma = [{'gamma': 2 ** i} for i in range(-7, 3, 2)]
gamma.extend([{'gamma': 'auto'}])
svc_params = [{**d1, **d2} for d1, d2 in product(C, gamma)]

n_h = [{'hidden_layer_sizes': [n]} for n in
       [15, 50, 100, 500, 1000]]
# discarded a lot of intermediate (100 is not the best, but just in case to keep
# one a bit 'large'
m_solver = [{'solver': s} for s in ['lbfgs', 'adam']]
# discarded sgd
m_activations = [{'activation': s} for s in ['tanh', 'relu']]
# discarded identity and logistic
# m_activations = [{'activation': s} for s in ['identity', 'logistic', 'tanh', 'relu']]
mlp_params = [{**d1, **d2, **d3} for d1, d2, d3 in
              product(n_h, m_solver, m_activations)]

raf_params = [{'n_estimators': i, 'n_jobs': 12} for i in [50, 100, 250, 500]]
ada_params = [{'n_estimators': i} for i in [50, 100, 250, 500]]


debug_1_classifiers = {'RFR': (RandomForestRegressor, [{'n_jobs': 2}]),
                       'graham': (LinearRegression, [{}]),
                       'MLPR': (MLPRegressor, [{}, {}]),
                       'MLPC': (MLPClassifier, [{}, {}]),
                       }

exp_1_classifiers = {**graham,
                     **{'SVC': (SVC, svc_params),
                        'MLPC': (MLPClassifier, mlp_params),
                        'RFC': (RandomForestClassifier, raf_params),
                        'AdaBC': (AdaBoostClassifier, ada_params)}}

# exp_2_classifiers = OrderedDict({**graham,
exp_2_classifiers = OrderedDict({'RFC': (
    RandomForestClassifier, raf_params),
    'RFR': (
        RandomForestRegressor, raf_params),
    'AdaBR': (AdaBoostRegressor, ada_params),
    'AdaBC': (AdaBoostClassifier, ada_params),
    'MLPR': (MLPRegressor, mlp_params),
    'MLPC': (MLPClassifier, mlp_params),
    'LR': (LinearRegression, [{}]),
    'SVC': (SVC, svc_params),
    'SVR': (SVR, svc_params)})

exp_3_classifiers = OrderedDict({'RFR': (RandomForestRegressor, raf_params),
                                 'AdaBR': (AdaBoostRegressor, ada_params),
                                 'MLPR': (MLPRegressor, mlp_params),
                                 'LR': (LinearRegression, [{}]),
                                 'SVR': (SVR, svc_params)})
