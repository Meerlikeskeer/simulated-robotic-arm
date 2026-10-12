import json
from pathlib import Path

import numpy as np

from arm import Arm
from datasets import dataset_builder
from kinematics import joint_positions
from neural_network import (
    evaluate_positions,
    predict_positions,
    train_nonlinear_network,
)


def main():
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    arm = Arm((0.0, 0.0), (2.0, 1.0))
    train = dataset_builder(1000, arm, seed=10)
    test = dataset_builder(2000, arm, seed=20)
    a = np.linspace(-np.pi, np.pi, 81)
    t1, t2 = np.meshgrid(a, a)
    angles = np.column_stack((t1.ravel(), t2.ravel()))

    target = np.array([joint_positions(Arm(tuple(t), arm.lengths))[-1] for t in angles])
    rows = []
    for seed in (1, 2, 3):
        for periodic in (False, True):
            p = train_nonlinear_network(
                train[:, :2],
                train[:, 2:],
                0.01,
                1000,
                periodic=periodic,
                seed=seed,
                shuffle=True,
            )
            boundary = np.linspace(-np.pi, np.pi, 101)
            left = np.column_stack((np.full(101, -np.pi), boundary))
            right = left.copy()
            right[:, 0] = np.pi
            gap = np.linalg.norm(
                predict_positions(left, p) - predict_positions(right, p), axis=1
            ).max()
            row = dict(
                seed=seed,
                periodic=periodic,
                train=evaluate_positions(train[:, :2], train[:, 2:], p),
                test=evaluate_positions(test[:, :2], test[:, 2:], p),
                grid=evaluate_positions(angles, target, p),
                max_shoulder_gap=float(gap),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            if periodic and seed == 1:
                np.savez(
                    output / "periodic_model.npz",
                    weights1=p[0],
                    biases1=p[1],
                    weights2=p[2],
                    biases2=p[3],
                    lengths=np.array(arm.lengths),
                )
    (output / "comparison.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
