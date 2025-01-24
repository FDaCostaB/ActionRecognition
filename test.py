import numpy as np
import matplotlib.pyplot as plt


def lrSchedulerTriangular(epoch, factor, base_lr, decay):
    cycle = 1 + np.floor(epoch / decay)
    x = np.abs(epoch / (decay * 0.5) - 2 * cycle + 1)

    scale = factor ** (cycle - 1)  # Shrinks max_lr every cycle

    new_lr = base_lr + base_lr * max(0, (1 - x)) * scale
    return new_lr

# Parameters
factor = 0.33
base_lr = 0.0004
decay = 10
epochs = np.arange(0, 30)

# Compute learning rates
learning_rates = [lrSchedulerTriangular(epoch, factor, base_lr, decay) for epoch in epochs]

# Plot the learning rate schedule
plt.figure(figsize=(10, 6))
plt.plot(epochs, learning_rates, marker='o', label="Learning Rate")
plt.title("Triangular Learning Rate Scheduler")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.legend()
plt.show()
