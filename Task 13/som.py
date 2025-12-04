import numpy as np
import matplotlib.pyplot as plt
from utils import *
import matplotlib
matplotlib.use('TkAgg')  # Remove or change if live-plotting is not working


class SOM:

    def __init__(self, input_dim, num_neurons):
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.reset_weights()

    def reset_weights(self, d=None):
        # Sets all weights to reasonably scaled random values
        self.W = np.random.rand(self.num_neurons, self.input_dim)
        if d is not None:
            low = np.min(d, axis=0)
            high = np.max(d, axis=0)
            self.W = low + self.W * (high - low)

    def distance(self, a, b):
        # Euclidian distance of two points (e.g. two colors).
        return np.linalg.norm(a - b)

    def find_winner(self, x):
        # Find winning neuron and return its INDEX (not its position)
        ### YOUR CODE GOES HERE ###
        return 0 

    def train(self, inputs, num_epoch=50, alpha_s=0.05, alpha_f=0.01, lambda_s=None, lambda_f=1,
              live_plot=True, live_interval=5):
        # Trains the SOM network and plots the progress.

        # Live-plotting stuff
        if lambda_s is None:
            lambda_s = self.num_neurons // 3
        if live_plot:
            plt.ion()
            plot_som(model=self, live_plot=True)
        
        self.reset_weights(data)
        for ep in range(num_epoch):
            alpha_t = 0  # FIXME
            lambda_t = 0  # FIXME

            data_count = data.shape[0]
            for i in np.random.permutation(data_count):
                x = inputs[i]
                ### YOUR CODE GOES HERE ###


            # Print and plot at the end of the episode
            print('Ep {:3d}/{:3d}: alpha_t = {:.3f}, lambda_t = {:.3f}'.format(ep+1, num_epoch, alpha_t, lambda_t))
            if live_plot and ((ep+1) % live_interval == 0):
                plot_som(model=self, live_plot=True)

        # Live-plotting stuff
        plot_som(model=self)
        if live_plot:
            plt.ioff()


if __name__ == "__main__":
    data = prepare_data('input.dat')
    model = SOM(input_dim=data.shape[1], num_neurons=30)
    model.train(data, num_epoch=25, live_plot=True)

    # Comparison with simpler PCA method
    plot_pca()

    if plt.get_fignums():
        plt.show(block=True)
