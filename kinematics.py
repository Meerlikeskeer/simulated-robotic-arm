import math
import numpy as np


def joint_positions(theta1, theta2, l1, l2):
    elbow_x = l1 * math.cos(theta1)
    elbow_y = l1 * math.sin(theta1)
    tip_x = l2 * math.cos(theta1 + theta2) + elbow_x
    tip_y = l2 * math.sin(theta1 + theta2) + elbow_y

    return np.array([[0, 0], [elbow_x, elbow_y], [tip_x, tip_y]])
