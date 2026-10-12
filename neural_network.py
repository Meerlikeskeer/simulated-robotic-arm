import numpy as np


def encode_angles(inputs):
    """Map angle pairs to periodic features: sin1, cos1, sin2, cos2."""
    inputs = np.asarray(inputs)
    return np.stack(
        (
            np.sin(inputs[..., 0]),
            np.cos(inputs[..., 0]),
            np.sin(inputs[..., 1]),
            np.cos(inputs[..., 1]),
        ),
        axis=-1,
    )


def predict_positions(inputs, parameters):
    """Predict positions for rows of angle pairs without updating parameters."""
    weights1, biases1, weights2, biases2 = parameters
    if weights1.shape[1] == 4:
        inputs = encode_angles(inputs)
    hidden = np.tanh(inputs @ weights1.T + biases1)
    return hidden @ weights2.T + biases2


def evaluate_positions(inputs, targets, parameters):
    """Return positional errors in the same units as the link lengths."""
    predictions = predict_positions(inputs, parameters)
    errors = np.linalg.norm(predictions - targets, axis=1)
    return {
        "rms": float(np.sqrt(np.mean(errors**2))),
        "mean": float(np.mean(errors)),
        "max": float(np.max(errors)),
    }


def training_step(weight, bias, input, target, learning_rate):
    prediction = weight @ input + bias
    error = prediction - target

    weight_gradient = np.outer(error, input)
    bias_gradient = error

    new_weight = weight - learning_rate * weight_gradient
    new_bias = bias - learning_rate * bias_gradient

    return new_weight, new_bias


def forward_backpropagation(weights1, biases1, weights2, biases2, input, target):
    z = weights1 @ input + biases1
    hidden = np.tanh(z)
    prediction = weights2 @ hidden + biases2

    error = prediction - target

    d_prediction = error
    d_weights2 = np.outer(d_prediction, hidden)
    d_biases2 = d_prediction

    d_hidden = weights2.T @ d_prediction
    d_z = d_hidden * (1 - hidden**2)
    d_weights1 = np.outer(d_z, input)
    d_biases1 = d_z

    return d_weights1, d_biases1, d_weights2, d_biases2


def nonlinear_training_step(
    weights1, biases1, weights2, biases2, input, target, learning_rate
):

    d_weights1, d_biases1, d_weights2, d_biases2 = forward_backpropagation(
        weights1, biases1, weights2, biases2, input, target
    )
    new_weights1 = weights1 - learning_rate * d_weights1
    new_biases1 = biases1 - learning_rate * d_biases1
    new_weights2 = weights2 - learning_rate * d_weights2
    new_biases2 = biases2 - learning_rate * d_biases2

    return new_weights1, new_biases1, new_weights2, new_biases2


def train_linear_network(input_thetas, target_positions, learning_rate, epochs):
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


def train_nonlinear_network(
    input_thetas,
    target_positions,
    learning_rate,
    epochs,
    callback=None,
    periodic=False,
    seed=None,
    shuffle=False,
):
    hidden_size = 16
    rng = np.random.default_rng(seed)
    features = encode_angles(input_thetas) if periodic else np.asarray(input_thetas)
    weights1 = rng.normal(size=(hidden_size, features.shape[1])) * 0.01
    biases1 = np.zeros(hidden_size)
    weights2 = rng.normal(size=(2, hidden_size)) * 0.01
    biases2 = np.zeros(2)

    for epoch in range(epochs):
        order = rng.permutation(len(features)) if shuffle else np.arange(len(features))
        for index in order:
            inputs, target = features[index], target_positions[index]
            weights1, biases1, weights2, biases2 = nonlinear_training_step(
                weights1,
                biases1,
                weights2,
                biases2,
                inputs,
                target,
                learning_rate,
            )

        if callback is not None and (
            epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs
        ):
            callback(epoch + 1, weights1, biases1, weights2, biases2)

    return weights1, biases1, weights2, biases2
