import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----- Generate synthetic data -----
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 3.5 * x + 7 + np.random.randn(*x.shape) * 3

# ----- Gradient Descent Parameters -----
lr = 0.0005          # TODO: play with different values of learning rate
epochs = 200
w0, w1 = 0.0, 0.0
loss_history = []

# ----- Set up plot -----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plt.tight_layout(pad=3)

# Left: regression fit
ax1.scatter(x, y, color='blue', alpha=0.6, label='Data')
line, = ax1.plot([], [], color='red', lw=2)
ax1.set_xlim(0, 10)
ax1.set_ylim(min(y)-5, max(y)+5)
ax1.set_title("Fitting Line (Gradient Descent)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Right: loss curve
ax2.set_xlim(0, epochs)
ax2.set_ylim(0, 200)
ax2.set_title("Loss (MSE) Over Time")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Loss")
loss_line, = ax2.plot([], [], color='green')


# ----- Animation function -----
def update(frame):
    global w0, w1
    y_pred = w1 * x + w0

    error = y - y_pred             # TODO: vector of element-wise differences
    loss = np.mean(error**2)
    loss_history.append(loss)

    # Compute gradients
    dw0 = -2 * np.sum(error)       # TODO: calculate and fill in the gradient
    dw1 = -2 * np.sum(error * x)  # TODO: calculate and fill in the gradient

    # Update parameters
    w0 -= lr * dw0                 # TODO: use gradient descent to update the weights
    w1 -= lr * dw1           # TODO: use gradient descent to update the weights

    # Update line plot
    line.set_data(x, y_pred)
    ax1.set_title(f"Fitted points")

    # Update loss plot
    loss_line.set_data(range(len(loss_history)), loss_history)
    return line, loss_line


# ----- Create animation -----
ani = FuncAnimation(fig, update, frames=epochs, interval=50, blit=True, repeat=False)

plt.show()
