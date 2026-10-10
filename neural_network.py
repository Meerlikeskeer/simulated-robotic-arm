import numpy as np


def training_step(weight, bias, input, target, learning_rate):
    prediction = weight @ input + bias
    error = prediction - target

    weight_gradient = np.outer(error, input)
    bias_gradient = error

    new_weight = weight - learning_rate * weight_gradient
    new_bias = bias - learning_rate * bias_gradient

    return new_weight, new_bias


def train_network(input_thetas, target_positions, learning_rate, epochs):
    weight = np.array(
        [
            [2.0, -1.0],  # Weights for predicting x
            [0.0, 3.0],  # Weights for predicting y
        ]
    )
    bias = np.array([1.0, -1.0])

    for epoch in range(epochs):
        for input, target in zip(input_thetas, target_positions):
            weight, bias = training_step(weight, bias, input, target, learning_rate)

    return weight, bias
