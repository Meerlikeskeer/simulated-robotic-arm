import kinematics
import math
import numpy as np
import arm


def row_builder(arm: arm.Arm):
    positions = kinematics.joint_positions(arm)
    tip = positions[-1]
    row = np.array([arm.angles[0], arm.angles[1], tip[0], tip[1]])
    return row


def dataset_builder(num_samples: int, arm1: arm.Arm):
    dataset = []
    for _ in range(num_samples):
        arm_instance = arm.Arm(
            angles=(
                np.random.uniform(-math.pi, math.pi),
                np.random.uniform(-math.pi, math.pi),
            ),
            lengths=arm1.lengths,
        )
        row = row_builder(arm_instance)
        dataset.append(row)

    return np.array(dataset)
