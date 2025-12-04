import numpy as np
import matplotlib.pyplot as plt
import random
import os
import matplotlib
matplotlib.use('TkAgg')  # Remove or change if live-plotting is not working

palette = ['r', 'lime', 'b', 'y']
scatter_kwargs = {'cmap': matplotlib.colors.ListedColormap(palette), 'edgecolors': [0.4]*3, 'alpha': 0.8, 's': 40}

data, labels = None, None


def prepare_data(file_path):
    global data, labels
    raw = np.loadtxt(file_path)
    data = raw[:, :-1]
    labels = raw[:, -1]
    return data


def plot_som(model, live_plot=False):
    # Plot data, SOM network and projected points with classes
    plt.figure('SOM')
    plt.clf()
    fig, ax = plt.subplots(2, 1, num='SOM', gridspec_kw={'height_ratios': [4, 1]})
    fig.canvas.mpl_connect('key_press_event', keypress)

    ax[0].scatter(data[:, 0], data[:, 1], c=labels, **scatter_kwargs)  # data
    ax[0].plot(model.W[:, 0], model.W[:, 1], '-k')                                      # lines between neurons
    ax[0].scatter(model.W[:, 0], model.W[:, 1], marker='x', c='black', s=80, lw=2)      # 'X' at neurons
    # Projected data
    x_trans = np.array([model.find_winner(data[i]) for i in range(data.shape[0])])
    dummy_y = np.random.randn(data.shape[0])
    ax[1].plot([-1, model.num_neurons + 1], [0, 0], '--', c='gray')
    ax[1].scatter(x_trans, dummy_y, c=labels, **scatter_kwargs)
    ax[1].set_yticklabels([])

    fig.tight_layout()
    if live_plot:
        plt.show(block=False)
        plt.gcf().canvas.draw()


def plot_pca():
    # Compute and plot PCA method results
    fig, ax = plt.subplots(2, 1, num='PCA', gridspec_kw={'height_ratios': [4, 1]})
    fig.canvas.mpl_connect('key_press_event', keypress)
    fig.tight_layout()
    # PCA
    count = data.shape[0]  # number of data points
    mu = np.mean(data, axis=0).reshape((2, 1))
    norm_data = data.T - mu
    Q = (norm_data @ norm_data.T) / count
    lambdas, V = np.linalg.eig(Q)
    first_pca = V[:, np.argmax(np.abs(lambdas))]
    # Original data and first principal component
    ax[0].scatter(data[:, 0], data[:, 1], c=labels, **scatter_kwargs)
    line = mu + np.stack([-first_pca, first_pca]) * 2.5
    ax[0].plot(line[0], line[1], '--', c='gray')
    # Projected data
    x_trans = (data - mu.T) @ first_pca
    dummy_y = np.random.randn(data.shape[0])
    ax[1].plot([-2.5, 2.5], [0, 0], '--', c='gray')
    ax[1].scatter(x_trans, dummy_y, c=labels, **scatter_kwargs)
    ax[1].set_yticklabels([])


def keypress(e):
    if e.key in {'q', 'escape'}:
        os._exit(0)  # unclean exit, but exit() or sys.exit() won't work
    if e.key in {' ', 'enter'}:
        plt.close()  # skip blocking figures