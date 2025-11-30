import numpy as np
from utils import *
import time

class MultiLayerPerceptron:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # FIXME: initialize the weight matrices with small values
        self.W1 = np.random.randn(self.hidden_dim, self.input_dim + 1) * 0.1
        self.W2 = np.random.randn(self.output_dim, self.hidden_dim + 1) * 0.1
    def add_bias(self, x):
        return np.concatenate([x, [1]])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_prime(self, x):
        y = self.sigmoid(x)
        return y * (1 - y)

    def compute_output(self, x):
        # FIXME: compute the output (x already includes the bias)

        net_hid = self.W1 @ x
        h = self.sigmoid(net_hid)

        h_bias = self.add_bias(h)
        net_out = self.W2 @ h_bias
        
        return net_out

    def compute_accuracy(self, inputs, targets):
        correct = 0
        total = inputs.shape[1]

        for i in range(total):
            x = self.add_bias(inputs[:, i])
            y = self.compute_output(x)
            if np.argmax(y) == np.argmax(targets[:, i]):
                correct += 1

        return correct / total

    def evaluate(self, file_path):
        inputs, targets = load_data(file_path)

        total_error = 0
        total_samples = inputs.shape[1]

        for i in range(total_samples):
            x = self.add_bias(inputs[:, i])
            d = targets[:, i]
            
            y = self.compute_output(x) 

            total_error += np.linalg.norm(d - y) ** 2 

        avg_error = total_error / total_samples
        return avg_error

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
                
                net_hid = self.W1 @ x
                h = self.sigmoid(net_hid)
                h_bias = self.add_bias(h)

                net_out = self.W2 @ h_bias
                y = net_out
                
                d = d.reshape(self.output_dim, 1)
                y = y.reshape(self.output_dim, 1)

                E_train += np.linalg.norm(d - y) ** 2
                delta_out = (d - y)
                error_hidden = (self.W2[:, :self.hidden_dim]).T @ delta_out
                delta_hid = error_hidden * self.sigmoid_prime(net_hid.reshape(self.hidden_dim, 1))

                self.W2 = self.W2 + alpha * delta_out @ h_bias.reshape(1, self.hidden_dim + 1)
                self.W1 = self.W1 + alpha * delta_hid @ x.reshape(1, self.input_dim + 1)


            err_hist.append(E_train)
            acc_hist.append(self.compute_accuracy(inputs, targets))

            if (ep + 1) % 10 == 0:
                print(f'Epoch {ep+1:03d}: Train Acc={acc_hist[-1]*100:.1f}%, Train Error={E_train:.3f}')

        return err_hist, acc_hist
    


if __name__ == '__main__':
    start = time.time()
    FILE_NAME = "mlp_train.txt"
    EVAL_FILE_NAME = "eval.txt"

    train_inputs, train_targets = load_data(FILE_NAME)
    
    print(f"Number of training data: {train_inputs.shape[1]} samples.")
                
    BEST_HIDDEN_DIM = 900
    BEST_EPOCHS = 2400
    BEST_ALPHA_START = 0.01
    BEST_ALPHA_END = 0.001
    
    
    mlp = MultiLayerPerceptron(
        train_inputs.shape[0],
        hidden_dim = BEST_HIDDEN_DIM,
        output_dim = train_targets.shape[0]
    )
    
    err_hist_mlp, acc_hist_mlp = mlp.train(
        train_inputs,
        train_targets,
        num_epochs = BEST_EPOCHS,
        alpha_start = BEST_ALPHA_START,
        alpha_end = BEST_ALPHA_END
    )
    
    # plot_training_history(err_hist_mlp, acc_hist_mlp)
    # plot_decision_boundary(mlp, train_inputs, train_outputs, title=f"MLP Decision Boundary")

    end = time.time()
    print(f"{int((end - start) // 60)}:{(end - start) % 60:05.2f}")

    #print(mlp.evaluate(EVAL_FILE_NAME))
    print(f"Average error for evaluation file is {mlp.evaluate(EVAL_FILE_NAME)}")




