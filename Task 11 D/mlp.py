import numpy as np
from utils import *
import time

class MultiLayerPerceptron:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # FIXME: initialize the weight matrices with small values
        self.W1 = np.random.randn(self.hidden_dim, self.input_dim + 1) 
        self.W2 = np.random.randn(self.output_dim, self.hidden_dim + 1)
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

        total_squared_error = 0
        total_samples = inputs.shape[1]
        
        if total_samples == 0:
            return 0.0

        for i in range(total_samples):
            x = self.add_bias(inputs[:, i])
            d = targets[:, i]
            
            y = self.compute_output(x) 

            total_squared_error += np.linalg.norm(d - y) ** 2 

        avg_error = total_squared_error / total_samples
        return avg_error.item()

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
                delta_out = (d - y) * 1.0
                error_prop_to_hidden = (self.W2[:, :self.hidden_dim]).T @ delta_out
                delta_hid = error_prop_to_hidden * self.sigmoid_prime(net_hid.reshape(self.hidden_dim, 1))

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
    
    try:
        full_inputs, full_targets = load_data(FILE_NAME)
        
        final_inputs = full_inputs
        final_targets = full_targets
        
        print(f"Loaded ALL available data for final training: {final_inputs.shape[1]} samples.")
        
        train_inputs, train_targets, val_inputs, val_targets = split_data(full_inputs, full_targets, train_ratio=0.8)
        
    except FileNotFoundError:
        print(f"ERROR: '{FILE_NAME}' not found. Cannot proceed with final training.")
        raise
        
    
    BEST_HIDDEN_DIM = 1000
    BEST_EPOCHS = 2700
    BEST_ALPHA_START = 0.005
    BEST_ALPHA_END = 0.0001
    
    print("\nStarting final training of the optimized MLP model...")
    
    mlp = MultiLayerPerceptron(
        final_inputs.shape[0], 
        hidden_dim=BEST_HIDDEN_DIM, 
        output_dim=final_targets.shape[0]
    )
    
    err_hist_mlp, acc_hist_mlp = mlp.train(
        final_inputs, 
        final_targets, 
        num_epochs=BEST_EPOCHS, 
        alpha_start=BEST_ALPHA_START, 
        alpha_end=BEST_ALPHA_END
    )
    
    # plot_training_history(err_hist_mlp, acc_hist_mlp)
    # plot_decision_boundary(mlp, final_inputs, final_targets, title=f"MLP Decision Boundary")

    temp_train_file = "temp_train_eval.txt"
    data_out = np.concatenate((train_inputs.T, train_targets.T), axis=1)
    np.savetxt(temp_train_file, data_out)
    
    avg_train_error = mlp.evaluate(temp_train_file)
    print(f"\nAverage TRAINING Error (evaluate function check): {avg_train_error:.6f}")
    os.remove(temp_train_file)

    temp_val_file = "temp_val_eval.txt"
    data_out = np.concatenate((val_inputs.T, val_targets.T), axis=1)
    np.savetxt(temp_val_file, data_out)
    
    avg_validation_error = mlp.evaluate(temp_val_file)
    print(f"Average VALIDATION Error (evaluate function check): {avg_validation_error:.6f}")
    os.remove(temp_val_file)
    
    print("\nTraining and evaluation checks complete. The script is ready for submission.")
    plt.show(block=True)
    end = time.time()
    print(f"{int((end - start) // 60)}:{(end - start) % 60:05.2f}")
