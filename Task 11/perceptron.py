import numpy as np

from utils import *


class SingleLayerPerceptron:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # FIXME: initialize the weight matrix with small values
        self.W = np.random.randn(self.output_dim, self.input_dim + 1)

    def add_bias(self, x):
        return np.concatenate([x, [1]])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_output(self, x):
        # FIXME: compute the output (x already includes the bias)
        return self.sigmoid(self.W @ x)

    def compute_accuracy(self, inputs, targets):
        correct = 0
        total = inputs.shape[1]

        for i in range(total):
            x = self.add_bias(inputs[:, i])
            y = self.compute_output(x)
            if np.argmax(y) == np.argmax(targets[:, i]):
                correct += 1

        return correct / total

    def train(self, inputs, targets, num_epochs=200, alpha_start=0.5, alpha_end=0.01):
        count = inputs.shape[1]
        err_hist = []
        acc_hist = []

        for ep in range(num_epochs):
            # FIXME: slowly decrease the learning rate (starts at alpha_start, ends at alpha_end)
            alpha = alpha_start -  ep * (alpha_start - alpha_end) / num_epochs 

            E_train = 0

            for i in np.random.permutation(count):
                x = self.add_bias(inputs[:, i])
                d = targets[:, i]

                # FIXME: compute the output, track the training error and update the weights
                y = self.compute_output(x)
                E_train += np.linalg.norm(d - y) ** 2
                delta = (d - y) * y * (1 - y)
                self.W = self.W + alpha * (np.outer(delta,x))

            err_hist.append(E_train)
            acc_hist.append(self.compute_accuracy(inputs, targets))

            if (ep + 1) % 10 == 0:
                print(f'Epoch {ep+1:03d}: Train Acc={acc_hist[-1]*100:.1f}%, Train Error={E_train:.3f}')

        return err_hist, acc_hist


class MultiLayerPerceptron:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # FIXME: initialize the weight matrices with small values
        self.W1 = 9
        self.W2 = 9

    def add_bias(self, x):
        return np.concatenate([x, [1]])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_output(self, x):
        # FIXME: compute the output (x already includes the bias)
        return 9

    def compute_accuracy(self, inputs, targets):
        correct = 0
        total = inputs.shape[1]

        for i in range(total):
            x = self.add_bias(inputs[:, i])
            y = self.compute_output(x)
            if np.argmax(y) == np.argmax(targets[:, i]):
                correct += 1

        return correct / total

    def train(self, inputs, targets, num_epochs=200, alpha_start=0.5, alpha_end=0.01):
        count = inputs.shape[1]
        denom = max(1, num_epochs - 1)
        err_hist = []
        acc_hist = []

        for ep in range(num_epochs):
            alpha = alpha_start * (alpha_end / alpha_start) ** (ep / denom)
            E_train = 0

            for i in np.random.permutation(count):
                x = self.add_bias(inputs[:, i])
                d = targets[:, i]

                # FIXME: compute the output, track the training error and update the weights
                y = 9
                E_train += 9
                self.W += 9

            err_hist.append(E_train)
            acc_hist.append(self.compute_accuracy(inputs, targets))

            if (ep + 1) % 10 == 0:
                print(f'Epoch {ep+1:03d}: Train Acc={acc_hist[-1]*100:.1f}%, Train Error={E_train:.3f}')

        return err_hist, acc_hist


if __name__ == '__main__':
    DATASET = 'linear'  # linear or 'circles'
    inputs, targets = prepare_data(kind=DATASET, num_points=1500, noise=0.05, seed=42)

    print("Training Single Layer Perceptron")
    slp = SingleLayerPerceptron(inputs.shape[0], targets.shape[0])
    err_hist_slp, acc_hist_slp = slp.train(inputs, targets, num_epochs=150)
    plot_training_history(err_hist_slp, acc_hist_slp)
    plot_decision_boundary(slp, inputs, targets, title=f"SLP Decision Boundary ({DATASET})")

    # print("\nTraining Multi Layer Perceptron")
    # mlp = MultiLayerPerceptron(inputs.shape[0], hidden_dim=5, output_dim=targets.shape[0])
    # err_hist_mlp, acc_hist_mlp = mlp.train(inputs, targets, num_epochs=150)
    # plot_training_history(err_hist_mlp, acc_hist_mlp)
    # plot_decision_boundary(mlp, inputs, targets, title=f"MLP Decision Boundary ({DATASET})")
