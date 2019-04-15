from collections import OrderedDict
from itertools import product

from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

train_attrs = ['eps', 'p2b', 'p2e', 'p2r', 'div2price',
               'divpayoutratio', 'roe', 'roic', 'roa', 'assetturnover',
               'invturnonver', 'profitmargin', 'debtratio', 'ebittointerestex',
               'wc', 'wc2a', 'currentratio', 'revenue', 'epsgrowth', 'bvps']

reg_classifiers = ['LR', 'MLPR', 'SVR', 'AdaBR', 'DTR', 'RFR']
cat_classifiers = ['MLPC', 'SVC', 'AdaBC', 'GBC', 'DTC', 'RFC']

# SVM params
C = [{'C': 2 ** i} for i in range(-5, 15, 2)]
gamma = [{'gamma': 2 ** i} for i in range(-7, 3, 2)]
gamma.extend([{'gamma': 'auto'}])
svc_params = [{**d1, **d2} for d1, d2 in product(C, gamma)]

# Neural network params
n_h = [{'hidden_layer_sizes': [n]} for n in
       [15, 50, 100, 500, 1000]]
# discarded a lot of intermediate (100 is not the best, but just in case to keep
# one a bit 'large'

m_solver = [{'solver': s} for s in ['lbfgs', 'adam']]
# discarded sgd

m_activations = [{'activation': s} for s in ['tanh', 'relu']]
# discarded identity and logistic

mlp_params = [{**d1, **d2, **d3} for d1, d2, d3 in
              product(n_h, m_solver, m_activations)]

# Random forst params
raf_params = [{'n_estimators': i, 'n_jobs': 12} for i in [50, 100, 250, 500]]

# AdaBoost params
ada_params = [{'n_estimators': i} for i in [50, 100, 250, 500]]

# Models per experiment to be used ==========================


# Just for debugging
debug = {'debug': (LinearRegression, [{}])}
debug_1_classifiers = {'RFR': (RandomForestRegressor, [{'n_jobs': 2}]),
                       'graham': (LinearRegression, [{}]),
                       'MLPR': (MLPRegressor, [{}, {}]),
                       'MLPC': (MLPClassifier, [{}, {}]),
                       }

# Experiment 1
exp_1_classifiers = {'graham': (LinearRegression, [{}]),
                     'SVC': (SVC, svc_params),
                     'MLPC': (MLPClassifier, mlp_params),
                     'RFC': (RandomForestClassifier, raf_params),
                     'AdaBC': (AdaBoostClassifier, ada_params)}

# Experiment 2
exp_2_classifiers = OrderedDict({'RFC': (RandomForestClassifier, raf_params),
                                 'RFR': (RandomForestRegressor, raf_params),
                                 'AdaBR': (AdaBoostRegressor, ada_params),
                                 'AdaBC': (AdaBoostClassifier, ada_params),
                                 'MLPR': (MLPRegressor, mlp_params),
                                 'MLPC': (MLPClassifier, mlp_params),
                                 'LR': (LinearRegression, [{}]),
                                 'SVC': (SVC, svc_params),
                                 'SVR': (SVR, svc_params)})
# Experiment 3
exp_3_classifiers = OrderedDict({'RFR': (RandomForestRegressor, raf_params),
                                 'AdaBR': (AdaBoostRegressor, ada_params),
                                 'MLPR': (MLPRegressor, mlp_params),
                                 'LR': (LinearRegression, [{}]),
                                 'SVR': (SVR, svc_params)})
