import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self):
        self.weight_matrix = np.random.randn()
        self.bias = np.random.randn()

    def sigmoid(self, x: float):
        return 1 / (1 + np.exp(-x))

    def squared_error(self, y_true: float, y_pred: float):
        return (y_true - y_pred) ** 2

    def output_function(self, x):
        return self.sigmoid(self.weight_matrix * x + self.bias)

    def update(self, x, y_true: float, y_pred: float, eta):
        dE_dy_pred = self.dE_dy_pred(y_true, y_pred)
        dy_pred_dw = self.dy_pred_dw(x, y_pred)
        dy_pred_db = self.dy_pred_db(y_pred)

        dE_dw = dE_dy_pred * dy_pred_dw
        dE_db = dE_dy_pred * dy_pred_db

        self.weight_matrix -= eta * dE_dw
        self.bias -= eta * dE_db

    def dE_dy_pred(self, y_true, y_pred):
        return -2 * (y_true - y_pred)

    def dy_pred_dw(self, x, y_pred):
        return y_pred * (1 - y_pred) * x

    def dy_pred_db(self, y_pred):
        return y_pred * (1 - y_pred)


def main():
    data = np.loadtxt("1d_classification_single_neuron.csv", delimiter=",")
    inputs = data[:, 0]
    labels = data[:, 1]

    neuron = Neuron()
    learning_rate = 0.01
    n_iterations = 100_000
    n_samples = len(inputs)
    errors = []

    rng = np.random.default_rng()
    for i in range(n_iterations):
        idx = rng.integers(n_samples)
        x = inputs[idx]
        y_true = labels[idx]
        y_pred = neuron.output_function(x)
        error = neuron.squared_error(y_true, y_pred)
        errors.append(error)
        neuron.update(x, y_true, y_pred, learning_rate)

    y_pred_final = [neuron.output_function(x) for x in inputs]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(inputs, labels, label="True Labels", color="blue", alpha=0.6)
    plt.scatter(
        inputs,
        y_pred_final,
        label="Neuron Prediction (after training)",
        color="red",
        alpha=0.6,
    )
    plt.xlabel("Input (X)")
    plt.ylabel("Output")
    plt.legend()
    plt.title("Predictions after Training")

    plt.subplot(1, 2, 2)
    plt.plot(errors, color="green", alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Squared Error")
    plt.title("Prediction Error during Training")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
