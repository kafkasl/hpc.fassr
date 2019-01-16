import argparse
import os
from time import time

import matplotlib
import pandas as pd
from pycompss.api.api import compss_wait_on

from data_managers.data_collector import get_data
from settings.basic import PROJECT_ROOT, DATE_FORMAT
from training import run_model

matplotlib.use('Agg')

if __name__ == '__main__':
    start_time = time()
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default=PROJECT_ROOT)

    args = parser.parse_args()

    if args.debug:
        resample_periods = ['3M']
        symbols_list_name = 'dow30'
        magic_number = 13
    else:
        resample_periods = ['1W', '2W', '1M', '3M']
        symbols_list_name = 'sp500'
        magic_number = 53

    money, choices = {}, {}

    for resample_period in resample_periods:
        dfn, dfz, attrs = get_data(resample_period=resample_period,
                                   symbols_list_name=symbols_list_name)

        datasets = {'normal': dfn, 'z-score': dfz}

        for d_key in datasets.keys():
            df = datasets[d_key]

            name = '%s_%s' % (d_key, resample_period)
            money[name], choices[name] = run_model(df, name, attrs,
                                                   magic_number)

    print("Waiting for the results.")
    money = compss_wait_on(money)
    choices = compss_wait_on(choices)

    exec_time = time()
    print("Execution time: %s" % (exec_time - start_time))

    choices_dict = {}
    for d_key in choices.keys():
        df_aux = pd.DataFrame(columns=list(choices[d_key].keys()),
                              index=pd.to_datetime([]))
        for clf_name in choices[d_key].keys():
            for date_str, topk, botk in choices[d_key][clf_name]:
                df_aux.loc[pd.to_datetime(date_str,
                                          format=DATE_FORMAT), clf_name] = len(
                    topk) + len(botk)

                choices_dict[d_key] = df_aux

    for name, res in choices_dict.items():
        plot = res.plot()
        fig = plot.get_figure()
        file_path = os.path.join(PROJECT_ROOT,
                                 "portfolio_size_%s.png" % (name))
        print("Saving file to %s" % file_path, end='')
        fig.savefig(file_path)
        print("Done.")

    for name, res in money.items():
        plot = res.plot()
        fig = plot.get_figure()
        file_path = os.path.join(PROJECT_ROOT, "results_%s.png" % (name))
        print("Saving file to %s" % file_path, end='')
        fig.savefig(file_path)
        print("Done.")

    total_time = time()
    print("Total time: %s" % (total_time - start_time))
