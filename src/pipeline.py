import argparse
from time import time

from pycompss.api.api import compss_wait_on

from data_managers.data_collector import get_data, get_prices
from settings.basic import PROJECT_ROOT
from training import explore_model

if __name__ == '__main__':
    start_time = time()
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default=PROJECT_ROOT)

    args = parser.parse_args()

    # datasets = ['normal', 'z-score']
    resample_periods = ['3M']
    symbols_list_name = 'dow30'
    magic_number = 13
    save_path = args.save_path

    if not args.debug:
        resample_periods = ['1W', '2W', '1M', '3M']
        symbols_list_name = 'sp500'
        magic_number = 53

    # params = itertools.product(resample_periods, datasets)


    results = {}
    accuracies = {}
    prices = get_prices(symbols_list_name='sp500',
                        resample_period='1D', only_prices=True)

    for resample_period in resample_periods:
        dfn, dfz, attrs = get_data(resample_period=resample_period,
                                   symbols_list_name=symbols_list_name)

        datasets = {'normal': dfn, 'z-score': dfz}

        for d_key in datasets.keys():
            df = datasets[d_key]

            dataset_name = '%s_%s' % (d_key, resample_period)
            portfolios, accs = explore_model(df=df, prices=prices,
                                             dataset_name=dataset_name,
                                             attrs=attrs,
                                             magic_number=magic_number,
                                             save_path=save_path)
            results[dataset_name] = portfolios
            accuracies[dataset_name] = accs

    results = compss_wait_on(results)
    accuracies = compss_wait_on(accuracies)
    print("Waiting for the results.")
    print(results)
    total_time = time()
    print("Total time: %.3f" % (total_time - start_time))
