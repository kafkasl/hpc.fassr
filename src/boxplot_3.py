from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
import matplotlib

from utils import load_obj

show_plot = False
cols = ["dataset", "period", "clf", "magic", "model_params", "k", "bot_thresh",
        "top_thresh", "mode", "trade_frequency", "start_trade", "final_trade",
        "time", "min", "max", "mean", "last"]


def add_legend(fig):
    # Finally, add a basic legend
    fig.text(0.8005, 0.135, '-', color='red', backgroundcolor='silver',
             weight='roman', size='medium')
    fig.text(0.825, 0.135, ' S&P 500 Index', color='black',
             weight='roman',
             size='x-small')

    fig.text(0.8005, 0.17, '*', color='white', backgroundcolor='silver',
             weight='roman', size='medium')
    fig.text(0.825, 0.17, ' Average Value', color='black', weight='roman',
             size='x-small')

def plot_by_k(results):
    # plot by model
    ks = [10, 25, 50]
    data = [results[pd.to_numeric(results.k) == k]['last'].values / 1e6 for k in ks]


    k_names = [20, 50, 100]

    fig, ax1 = plt.subplots(figsize=(5, 9))
    fig.canvas.set_window_title('Revenues per different portfolio sizes')
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
    ax1.ticklabel_format(axis='y', style='plain')
    ax1.set_title('Comparison of total revenues by portfolio sizes')
    ax1.set_xlabel('Portfolio size')
    ax1.set_ylabel('Total revenue in million U.S. dollars')
    ax1.axhline(y=276480 / 1e6, color='red', linestyle='--', alpha=0.4)

    # Now fill the boxes with desired colors
    boxColors = ['royalblue', 'royalblue']
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
    bottom = min([min(x) for x in data if len(x) > 0]) * 5

    ax1.set_ylim(-1, top)
    ax1.set_xticklabels(k_names, fontsize=10)

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


    add_legend(fig)
    plt.gcf().subplots_adjust(left=0.1)

    # plt.tight_layout()
    plt.savefig('ks', bbox_inches='tight')
    if show_plot:
        plt.show()


def plot_by_model(results):
    # plot by model
    models = ['SVR', 'RFR', 'MLPR', 'AdaBR']
    data = [results[results.clf == clf]['last'].values / 1e6 for clf in models]

    model_names = ['SVM', 'Random Forest', 'Neural Network',
                   'AdaBoost']

    fig, ax1 = plt.subplots(figsize=(7, 9))
    fig.canvas.set_window_title('Revenues per model')
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
    ax1.ticklabel_format(axis='y', style='plain')
    ax1.set_title('Comparison of total revenues for different models')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Total revenue in million U.S. dollars')
    ax1.axhline(y=276480 / 1e6, color='red', linestyle='--', alpha=0.4)

    # Now fill the boxes with desired colors
    boxColors = ['royalblue', 'royalblue']
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
    bottom = min([min(x) for x in data if len(x) > 0]) * 5

    ax1.set_ylim(-1, top)
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


    add_legend(fig)
    plt.gcf().subplots_adjust(left=0.1)

    # plt.tight_layout()
    plt.savefig('models', bbox_inches='tight')
    if show_plot:
        plt.show()

def to_df_col(pfs, name):
    index = [p._day_str for p in pfs]
    data = [(p.total_money - p.fees) / 1e6 for p in pfs]
    return pd.DataFrame(data=data, index=index, columns=[name])


def get_trends_df(results):
    best = results.groupby('clf')[['last']].max()
    worst = results.groupby('clf')[['last']].min()

    sp500 = pd.read_csv('../sp500.csv').set_index('Date')
    sp500.index = pd.to_datetime(sp500.index)
    sp500 = sp500[['Adj Close']].rename(columns={'Adj Close': 'S&P 500'})
    ratio = 100000 / sp500.iloc[0]

    b_found = []
    w_found = []
    b_trends = []
    w_trends = []
    import ipdb
    for file in glob('./*/clean_results_*'):
        print("Processing: %s" % file)
        result = load_obj(file[:-4])
        # prices =
        for i, (clf, price) in enumerate(best.itertuples()):
            print(" * searching for best: %s" % clf)
            trend = [r for r in result if
                     '%.1f' % r[1][-1].total_money == str(price)]
            if len(trend) > 0 and clf not in b_found:
                ipdb.set_trace()
                print('found! %s' % list(trend[0]))
                b_found.append(clf)
                b_trends.append(to_df_col(trend[0][1], clf))
        for i, (clf, price) in enumerate(worst.itertuples()):
            print(" * searching for worst: %s" % clf)
            trend = [r for r in result if
                     '%.1f' % r[1][-1].total_money == str(price)]
            if len(trend) > 0 and clf not in w_found:
                ipdb.set_trace()
                print('found! %s' % list(trend[0]))
                w_found.append(clf)
                w_trends.append(to_df_col(trend[0][1], clf))

    sptrend = (sp500 * ratio) / 1e6
    sptrend = sptrend.resample('1W').last()
    b_t = sptrend[sptrend.index.isin(b_trends[0].index)]
    b_trends.append(b_t)
    w_t = sptrend[sptrend.index.isin(w_trends[0].index)]
    w_trends.append(w_t)
    best_df = pd.concat(b_trends, axis=1).interpolate()
    worst_df = pd.concat(w_trends, axis=1).interpolate()

    best_df = best_df.reindex(sorted(best_df.columns), axis=1)
    worst_df = worst_df.reindex(sorted(worst_df.columns), axis=1)

    best_df.index = pd.to_datetime(best_df.index)
    worst_df.index = pd.to_datetime(worst_df.index)

    return best_df, worst_df


def plot_historicals(best, worst):
    names = ['SVM', 'RF', 'NN', 'AdaBoost', 'S&P 500']
    regs = ['SVR', 'RFR', 'MLPR', 'AdaBR', 'S&P 500']

    # plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    best.rename(columns={regs[i]: names[i] for i in range(len(regs))}).plot(ax=axes[0])
    worst.rename(columns={regs[i]: names[i] for i in range(len(regs))}).plot(ax=axes[1])

    [ax1.set_ylabel('Total revenue in million U.S. dollars') for ax1 in axes]
    axes[0].set_xlabel('Best')
    axes[1].set_xlabel('Worst')
    # [ax1.set_ylim(0, 2) for ax1 in axes]
    plt.tight_layout()
    plt.savefig('historicals')


if __name__ == '__main__':
    print("Loading ../../results/res3.csv")

    results = pd.read_csv('../../results/res3.csv', names=cols).sort_values(
        'last').drop(
        'time', 1).drop_duplicates()

    # r = r[(r.clf == 'AdaBC') | (r.clf == 'MLPC') | (r.clf == 'RFC') | (r.clf == 'SVC') | (r.clf == 'graham')]

    # results = r[(r.trade_frequency == 52) | (r.trade_frequency == 26)]

    plot_by_model(results)

    plot_by_k(results)

    best_df, worst_df = get_trends_df(results)
    plot_historicals(best_df, worst_df)
    # plot_by_frequency(results)
    # plot_by_training(results)
    # plot_by_threshold(results)
    # plot_by_mode(results)
