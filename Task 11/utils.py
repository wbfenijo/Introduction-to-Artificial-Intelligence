import matplotlib.pyplot as plt
import numpy as np

def one_hot(x, num_classes=3):
    b = np.zeros(num_classes)
    b[x] = 1.0
    return b

def prepare_data(num_points=1500, kind='linear', noise=0.05, seed=42):
    np.random.seed(seed)
    num_classes = 3

    if kind == 'linear':
        centers = [(-0.5, 0), (0, 0), (0.5, 0)]
        xs = []
        ys = []
        labs = []
        pts_per = num_points // num_classes

        for idx, (cx, cy) in enumerate(centers):
            x = np.random.randn(pts_per) * (0.05 + noise) + cx
            y = np.random.randn(pts_per) * (0.05 + noise) + cy
            xs.append(x)
            ys.append(y)
            for _ in range(len(x)):
                labs.append(one_hot(idx, num_classes))

        xs = np.concatenate(xs)[:num_points]
        ys = np.concatenate(ys)[:num_points]
        pts = np.vstack((xs, ys)).T
        pts = (pts - pts.min(0)) / (pts.max(0) - pts.min(0))
        labels = np.asarray(labs)[:num_points]
        return pts.T, labels.T

    elif kind == 'circles':
        pts_per = num_points // num_classes
        xs = []
        ys = []
        labs = []

        for c in range(num_classes):
            r = 0.1 + 0.25 * c
            theta = np.random.rand(pts_per) * 2 * np.pi
            x = r * np.cos(theta) + np.random.randn(pts_per) * noise
            y = r * np.sin(theta) + np.random.randn(pts_per) * noise
            xs.append(x)
            ys.append(y)
            for _ in range(len(x)):
                labs.append(one_hot(c, num_classes))

        xs = np.concatenate(xs)[:num_points]
        ys = np.concatenate(ys)[:num_points]
        pts = np.vstack((xs, ys)).T
        pts = (pts - pts.min(0)) / (pts.max(0) - pts.min(0))
        labels = np.asarray(labs)[:num_points]
        return pts.T, labels.T

    else:
        raise ValueError("Unknown dataset kind")

# ---------------- plotting ----------------

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
