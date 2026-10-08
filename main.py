import math
import numpy as np


def jointPositions(theta1, theta2, l1, l2):
    elbow_x = l1 * math.cos(theta1)
    elbow_y = l1 * math.sin(theta1)
    tip_x = l2 * math.cos(theta1 + theta2) + elbow_x
    tip_y = l2 * math.sin(theta1 + theta2) + elbow_y

    return np.array([[0, 0], [elbow_x, elbow_y], [tip_x, tip_y]])


def test_jointPositions():

    l1 = 1.0
    l2 = 1.0
    theta1 = math.pi / 4
    theta2 = math.pi / 4
    expected = np.array(
        [
            [0, 0],
            [0.70710678, 0.70710678],
            [0.70710678, 1.70710678],
        ]
    )
    # Each case: (l1, l2, theta1, theta2, expected_positions)
    test_cases = [
        (
            1.0,
            1.0,
            0.0,
            0.0,
            [[0, 0], [1, 0], [2, 0]],
        ),
        (
            1.0,
            1.0,
            0.0,
            math.pi / 2,
            [[0, 0], [1, 0], [1, 1]],
        ),
        (
            2.0,
            1.0,
            0.0,
            math.pi,
            [[0, 0], [2, 0], [1, 0]],
        ),
        (
            1.0,
            1.0,
            math.pi / 2,
            0.0,
            [[0, 0], [0, 1], [0, 2]],
        ),
    ]
    for l1, l2, theta1, theta2, expected in test_cases:
        actual = jointPositions(theta1, theta2, l1, l2)
        expected = np.array(expected)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
    print("All tests passed")


def main() -> None:
    test_jointPositions()


if __name__ == "__main__":
    main()
