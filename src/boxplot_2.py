from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from utils import load_obj

show_plot = False
cols = ["dataset", "period", "clf", "magic", "model_params", "k", "bot_thresh",
        "top_thresh", "mode", "trade_frequency", "start_trade", "final_trade",
        "time", "min", "max", "mean", "last"]

regressors = ['SVR', 'RFR', 'MLPR', 'AdaBR']

classifiers = ['SVC', 'RFC', 'MLPC', 'AdaBC']


def add_legend(fig):
    # Finally, add a basic legend
    fig.text(0.8005, 0.115, '-', color='red', backgroundcolor='silver',
             weight='roman', size='medium')
    fig.text(0.817, 0.115, ' S&P 500 Index returns', color='black',
             weight='roman',
             size='x-small')

    fig.text(0.8005, 0.165, '*', color='white', backgroundcolor='silver',
             weight='roman', size='medium')
    fig.text(0.815, 0.165, ' Average Value', color='black', weight='roman',
             size='x-small')


def plot_by_model(results):
    # plot by model
    models = ['SVC', 'SVR', 'RFC', 'RFR', 'MLPC', 'MLPR',
              'AdaBC',
              'AdaBR']
    data = [results[results.clf == clf]['last'].values / 1e6 for clf in models]

    model_names = ['', 'SVM', '', 'Random Forest', '', 'Neural Network', '',
                   'AdaBoost']

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('Returns per model')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    plt.setp(bp['medians'], color='black')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Comparison of total returns for different models')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Total returns in million U.S. dollars')
    ax1.axhline(y=276480 / 1e6, color='red', linestyle='--', alpha=0.4)
    # ax1.get_yaxis().set_major_formatter(
    #     matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    # Now fill the boxes with desired colors
    boxColors = ['darkkhaki', 'royalblue']
    numBoxes = len(data)
    medians = list(range(numBoxes))
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        k = i % 2
        boxCoords = np.column_stack([boxX, boxY])
        boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
        ax1.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            # ax1.plot(medianX, medianY, 'k')
            medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot([np.average(med.get_xdata())], [np.average(data[i])],
                 color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, numBoxes + 0.5)
    top = max([max(x) for x in data if len(x) > 0]) * 1.1
    bottom = min([min(x) for x in data if len(x) > 0]) * 1.5
    ax1.set_ylim(bottom, top)
    ax1.set_xticks([1.5, 1.5, 3.5, 3.5, 5.5, 5.5, 7.5, 7.5])
    ax1.set_xticklabels(model_names, fontsize=10)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(numBoxes) + 1
    upperLabels = [str(np.round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(numBoxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], top - (top * 0.05), upperLabels[tick],
                 horizontalalignment='center', size='x-small',
                 weight=weights[k],
                 color=boxColors[k])

    # Finally, add a basic legend
    # fig.text(0.80, 0.12, 'Classification',
    #          backgroundcolor=boxColors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.09, 'Regression',
    #          backgroundcolor=boxColors[1],
    #          color='white', weight='roman', size='x-small')
    fig.text(0.63, 0.15, ' ', color='white', backgroundcolor=boxColors[0],
             weight='roman', size='medium')
    fig.text(0.64, 0.15, ' Classification models', color='black',
             weight='roman',
             size='x-small')

    fig.text(0.63, 0.10, ' ', backgroundcolor=boxColors[1],
             weight='roman', size='medium')
    fig.text(0.64, 0.10, ' Regression models', color='black', weight='roman',
             size='x-small')

    fig.text(0.8005, 0.15, '*', color='white', backgroundcolor='silver',
             weight='roman', size='medium')
    fig.text(0.815, 0.15, ' Average Value', color='black', weight='roman',
             size='x-small')

    fig.text(0.8005, 0.10, '-', color='red', backgroundcolor='silver',
             weight='roman', size='medium')
    fig.text(0.815, 0.10, ' S&P 500 Index returns', color='black',
             weight='roman',
             size='x-small')

    plt.savefig('models', bbox_inches='tight')
    if show_plot:
        plt.show()


def to_df_col(pfs, name):
    index = [p._day_str for p in pfs]
    data = [(p.total_money - p.fees) / 1e6 for p in pfs]
    return pd.DataFrame(data=data, index=index, columns=[name])


def get_trends_df(results):
    best = results.groupby('clf')[['last']].max()
    sp500 = pd.read_csv('../sp500.csv').set_index('Date')
    sp500.index = pd.to_datetime(sp500.index)
    sp500 = sp500[['Adj Close']].rename(columns={'Adj Close': 'S&P 500'})
    ratio = 100000 / sp500.iloc[0]

    found = []
    trends = []
    for file in glob('./*/clean_results_*'):
        print("Processing: %s" % file)
        result = load_obj(file[:-4])
        # prices =
        for i, (clf, price) in enumerate(best.itertuples()):
            print(" * searching: %s" % clf)
            trend = [r for r in result if
                     '%.1f' % r[1][-1].total_money == str(price)]
            if len(trend) > 0 and clf not in found:
                print('found!')
                found.append(clf)
                trends.append(to_df_col(trend[0][1], clf))
                # res.extend(
                #     [r for r in result if '%.1f' % r[1][-1].total_money == str(price)])

    sptrend = (sp500 * ratio) / 1e6
    sptrend = sptrend.resample('1W').last()
    sptrend = sptrend[sptrend.index.isin(trends[0].index)]
    trends.append(sptrend)
    df = pd.concat(trends, axis=1).interpolate()

    df.index = pd.to_datetime(df.index)

    return df


def plot_historicals(df):
    names = ['SVM', 'RF', 'NN', 'AdaBoost', 'S&P 500']
    regs = ['SVR', 'RFR', 'MLPR', 'AdaBR', 'S&P 500']
    clas = ['SVC', 'RFC', 'MLPC', 'AdaBC', 'S&P 500']
    # plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    df[clas].rename(
        columns={clas[i]: names[i] for i in range(len(regs))}).plot(ax=axes[0])
    df[regs].rename(
        columns={regs[i]: names[i] for i in range(len(regs))}).plot(ax=axes[1])

    [ax1.set_ylabel('Total returns in million U.S. dollars') for ax1 in axes]
    axes[0].set_xlabel('Classification')
    axes[1].set_xlabel('Regression')
    [ax1.set_ylim(0, 2) for ax1 in axes]
    plt.tight_layout()
    plt.savefig('historicals')


if __name__ == '__main__':
    print("Loading ../res2.csv")

    results = pd.read_csv('../res2.csv', names=cols).sort_values(
        'last').drop(
        'time', 1).drop_duplicates()

    plot_by_model(results)
    # df = get_trends_df(results)
    # plot_historicals(df)
    # plot_by_dataset(results)
    # plot_by_frequency(results)
    # plot_by_training(results)
    # plot_by_threshold(results)
    # plot_by_mode(results)
