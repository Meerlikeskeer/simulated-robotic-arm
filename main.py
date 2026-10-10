import datasets
import visualization
import arm
import neural_network
import numpy as np


def main() -> None:
    # arm1 = arm.Arm(angles=(0.0, 0.5), lengths=(1.0, 1.0))
    # visualization.interactive_arm(arm1)
    # arm1 = arm.Arm(angles=(0.0, 0.0), lengths=(2.0, 1.0))
    # dataset = datasets.dataset_builder(1000, arm1)
    # visualization.plot_datasets(dataset)
    inputs = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    targets = np.array(
        [
            [1.0, -2.0],
            [3.0, -1.0],
            [0.0, 1.0],
            [2.0, 2.0],
        ]
    )
    weight, bias = neural_network.train_network(
        input_thetas=inputs, target_positions=targets, learning_rate=0.01, epochs=3000
    )
    print(f"Trained weight: {weight}, bias: {bias}")


if __name__ == "__main__":
    main()
