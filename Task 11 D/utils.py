import numpy as np
import matplotlib.pyplot as plt
import os



def press_to_quit(e):
    if e.key in {'q', 'escape'}:
        os._exit(0) # unclean exit, but exit() or sys.exit() won't work
    if e.key in {' ', 'enter'}:
        plt.close() # skip blocking figures


def show_history(history, block=True):
    fig = plt.figure(num='Training history')
    fig.canvas.mpl_connect('key_press_event', press_to_quit)

    plt.title('Loss per epoch')
    plt.plot(history.history['loss'], '-b', label='training loss')
    try:
        plt.plot(history.history['val_loss'], '-r', label='validation loss')
    except KeyError:
        pass
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlim(left=-1); plt.ylim(bottom=-0.01)

    plt.tight_layout()
    plt.show(block=block)


def show_data(X, y, predicted=None, s=30, block=True):
    plt.figure(num='Data', figsize=(9,9)).canvas.mpl_connect('key_press_event', press_to_quit)
    
    if predicted is not None:
        predicted = np.asarray(predicted).flatten()
        plt.subplot(2,1,2)
        plt.title('Predicted')
        plt.scatter(X[:, 0], X[:, 1],
                    c=predicted, cmap='coolwarm',
                    s=10 + s * np.maximum(0, predicted))
        
        plt.subplot(2,1,1)
        plt.title('Original')
    y = np.asarray(predicted).flatten()
    plt.scatter(X[:, 0], X[:, 1],
                c=y, cmap='coolwarm',
                s=10 + s * np.maximum(0, y))
    plt.tight_layout()
    
    plt.show(block=block)


def plot_decision_boundary(model, inputs, targets, grid_res=300, padding=0.05, title=None):
    x_min = inputs[0].min()
    x_max = inputs[0].max()
    y_min = inputs[1].min()
    y_max = inputs[1].max()
    dx = max(1e-6, x_max - x_min)
    dy = max(1e-6, y_max - y_min)

    x_min -= padding * dx
    x_max += padding * dx
    y_min -= padding * dy
    y_max += padding * dy

    xs = np.linspace(x_min, x_max, grid_res)
    ys = np.linspace(y_min, y_max, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.vstack([XX.ravel(), YY.ravel()])

    ones_row = np.ones((1, grid.shape[1]))
    grid_bias = np.vstack([grid, ones_row])

    Y = np.zeros((model.output_dim, grid_bias.shape[1]))
    for i in range(grid_bias.shape[1]):
        Y[:, i] = model.compute_output(grid_bias[:, i])

    preds = np.argmax(Y, axis=0).reshape(XX.shape)

    levels = np.arange(-0.5, model.output_dim + 0.5, 1)
    plt.contourf(XX, YY, preds, levels=levels, cmap=plt.get_cmap('tab10'), alpha=0.25)
    plt.scatter(inputs[0], inputs[1], c=np.argmax(targets, axis=0), cmap='tab10', s=8, edgecolor='k')
    if title:
        plt.title(title)
    plt.show()


def plot_training_history(err_hist, acc_hist):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(err_hist, '-r')
    plt.xlabel('Epoch')
    plt.ylabel('Training Error')
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(np.array(acc_hist)*100, '-b')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy [%]')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    
def load_data(file_path):
    data = np.loadtxt(file_path)
    inputs = data[:, :2].T 
    targets = data[:, 2].reshape(1, -1) 
    return inputs, targets

def split_data(inputs, targets, train_ratio=0.8, seed=42):
    np.random.seed(seed) 
    num_samples = inputs.shape[1]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_point = int(num_samples * train_ratio)
    train_inputs = inputs[:, indices[:split_point]]
    train_targets = targets[:, indices[:split_point]]
    val_inputs = inputs[:, indices[split_point:]]
    val_targets = targets[:, indices[split_point:]]
    return train_inputs, train_targets, val_inputs, val_targets
