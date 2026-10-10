import math
import numpy as np

from arm import Arm
from kinematics import joint_positions


def test_joint_positions():

    # Each case: (arm configuration, expected positions)
    test_cases = [
        (
            Arm(angles=(0.0, 0.0), lengths=(1.0, 1.0)),
            [[0, 0], [1, 0], [2, 0]],
        ),
        (
            Arm(angles=(0.0, math.pi / 2), lengths=(1.0, 1.0)),
            [[0, 0], [1, 0], [1, 1]],
        ),
        (
            Arm(angles=(0.0, math.pi), lengths=(2.0, 1.0)),
            [[0, 0], [2, 0], [1, 0]],
        ),
        (
            Arm(angles=(math.pi / 2, 0.0), lengths=(1.0, 1.0)),
            [[0, 0], [0, 1], [0, 2]],
        ),
    ]
    for arm, expected in test_cases:
        actual = joint_positions(arm)
        expected = np.array(expected)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
    print("All tests passed")
