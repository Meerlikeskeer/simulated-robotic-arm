import numpy as np

from neural_network import encode_angles, forward_backpropagation, predict_positions


def test_periodic_encoding():
    angles = np.array([[-np.pi, 0.5], [np.pi, 0.5]])
    np.testing.assert_allclose(
        encode_angles(angles)[0], encode_angles(angles)[1], atol=1e-14
    )


def test_gradients_against_finite_differences():
    rng = np.random.default_rng(42)
    parameters = [
        rng.normal(scale=0.2, size=shape) for shape in ((4, 2), (4,), (2, 4), (2,))
    ]
    inputs = np.array([0.3, -0.7])
    target = np.array([0.8, -0.2])
    gradients = forward_backpropagation(*parameters, inputs, target)

    def loss():
        error = predict_positions(inputs[None, :], parameters)[0] - target
        return 0.5 * error @ error

    for parameter, gradient in zip(parameters, gradients):
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + 1e-6
            plus = loss()
            parameter[index] = original - 1e-6
            minus = loss()
            parameter[index] = original
            numerical[index] = (plus - minus) / 2e-6
        np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-8)
