import sys
import matplotlib


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

show_plot = False
cols = ["dataset", "period", "clf", "magic", "model_params", "k", "bot_thresh",
        "top_thresh", "mode", "trade_frequency", "start_trade", "final_trade",
        "time", "min", "max", "mean", "last"]

exp = sys.argv[1]

rf = pd.read_csv('rf_%s.csv' % exp).sort_values('time')
svm = pd.read_csv('svm_%s.csv' % exp).sort_values('time')
mlp = pd.read_csv('mlp_%s.csv' % exp).sort_values('time')
ada = pd.read_csv('ada_%s.csv' % exp).sort_values('time')

res = pd.concat([rf, svm, mlp, ada], axis=0)


def add_legend(fig):

    # Finally, add a basic legend
    fig.text(0.8005, 0.165, '*', color='white', backgroundcolor='silver',
             weight='roman', size='medium')
    fig.text(0.815, 0.165, ' Average Value', color='black', weight='roman',
             size='x-small')

def plot_by_estimators(results, x_lab, param, fname, title):
    # plot by model
    estimators = sorted(set(results[param]))
    data = [results[results[param]==e]['time'].values for e in estimators]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('Execution times')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5, meanline=False)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    plt.setp(bp['medians'], color='black')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    # ax1.get_yaxis().set_major_formatter(
    #     # matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title(title)
    ax1.set_xlabel(x_lab)
    ax1.set_ylabel('Execution time (s)')


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
    bottom = min([min(x) for x in data if len(x) > 0])
    bottom = min(bottom - bottom*0.1, 0.05)
    ax1.set_ylim(bottom, top)
    # ax1.set_xticks([1, 2, 3.5, 3.5, 5.5, 5.5, 7.5, 7.5, 9.5, 9.5])
    ax1.set_xticklabels(estimators, fontsize=10)

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
    # fig.text(0.8005, 0.135, ' ', color='white', backgroundcolor=boxColors[0],
    #          weight='roman', size='medium')
    # fig.text(0.81, 0.135, ' Classification models', color='black', weight='roman',
    #          size='x-small')
    #
    # fig.text(0.8005, 0.09, ' ', backgroundcolor=boxColors[1],
    #          weight='roman', size='medium')
    # fig.text(0.81, 0.09, ' Regression models', color='black', weight='roman',
    #          size='x-small')

    add_legend(fig)
    plt.savefig('%s' % (fname), bbox_inches='tight')
    if show_plot:
        plt.show()

# RF
plot_by_estimators(rf, 'Number of trees', 'estimators', 'rf_estimators_%s' % exp, 'Execution times for different numbers of trees in random forests')
# Ada
plot_by_estimators(ada, 'Number of estimators', 'estimators', 'ada_estimators_%s' % exp, 'Execution times for different number of estimators in AdaBoost')
# Neural nets
plot_by_estimators(mlp, 'Number of neurons', 'size', 'mlp_size_%s' % exp, 'Execution times for different sizes of the hidden layer in neural networks')
plot_by_estimators(mlp, 'Solver', 'solver', 'mlp_solver_%s' % exp, 'Execution times for different solvers of the neural network')
plot_by_estimators(mlp, 'Activation function', 'activation', 'mlp_activation_%s' % exp, 'Execution times for different activation functions for the neural network')
# SVM
# TOO FEW VALUES:
plot_by_estimators(svm, 'C', 'C', 'svc_c_%s' % exp, 'Execution times for differnt values of C for the SVMs')
plot_by_estimators(svm, 'Gamma', 'gamma', 'svc_gamma_%s' % exp, 'Execution times for different gamma values for the SVMs')
