import argparse
import os

from pycompss.api.api import compss_wait_on

from data_managers.data_collector import get_data
from settings.basic import PROJECT_ROOT
from training import run_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()

    if args.debug:
        resample_periods = ['3M']
        symbols_list_name = 'dow30'
        magic_number = 13
    else:
        resample_periods = ['1W', '2W', '1M', '3M']
        symbols_list_name = 'sp500'
        magic_number = 53

    results = {}

    for resample_period in resample_periods:
        dfn, dfz, attrs = get_data(resample_period=resample_period,
                                   symbols_list_name=symbols_list_name)

        datasets = {'normal': dfn, 'z-score': dfz}

        for d_key in datasets.keys():
            df = datasets[d_key]

            name = '%s_%s' % (d_key, resample_period)
            results[name] = run_model(df, name=name, attrs=attrs, magic_number=magic_number)

    print("Waiting for the results.")
    results = compss_wait_on(results)

    for name, res in results.items():
        plot = res.plot()
        fig = plot.get_figure()
        file_path = os.path.join(PROJECT_ROOT, "results_%s.png" % (name))
        print("Saving file to %s" % file_path, end='')
        fig.savefig(file_path)
        print("Done.")
