import numpy as np

import arm
import datasets
import kinematics
import neural_network
import visualization


def training_demo() -> None:
    # arm1 = arm.Arm(angles=(0.0, 0.5), lengths=(1.0, 1.0))
    # visualization.interactive_arm(arm1)

    arm1 = arm.Arm(angles=(0.0, 0.0), lengths=(2.0, 1.0))
    dataset = datasets.dataset_builder(1000, arm1)

    # Separate examples, never passed into training.
    test_dataset = datasets.dataset_builder(1000, arm1)

    # visualization.plot_datasets(dataset)

    inputs = dataset[:, :2]  # theta1, theta2
    targets = dataset[:, 2:]  # tip x, tip y
    parameters = visualization.watch_training(
        inputs,
        targets,
        learning_rate=0.01,
        epochs=3000,
    )

    for label, evaluation_data in (
        ("Training", dataset),
        ("Unseen test", test_dataset),
    ):
        metrics = neural_network.evaluate_positions(
            evaluation_data[:, :2], evaluation_data[:, 2:], parameters
        )
        print(f"{label} positional error ({len(evaluation_data)} examples):")
        print(f"  RMS: {metrics['rms']:.6f}")
        print(f"  Mean: {metrics['mean']:.6f}")
        print(f"  Maximum: {metrics['max']:.6f}")

    check_boundary_predictions(arm1, parameters)
    visualization.plot_test_errors(test_dataset, parameters)


def check_boundary_predictions(arm1, parameters):
    for theta2 in (0.0, 0.5, np.pi):
        boundary_inputs = np.array([[-np.pi, theta2], [np.pi, theta2]])
        predictions = neural_network.predict_positions(boundary_inputs, parameters)
        actual = np.array(
            [
                kinematics.joint_positions(
                    arm.Arm(angles=tuple(angles), lengths=arm1.lengths)
                )[-1]
                for angles in boundary_inputs
            ]
        )
        gap = np.linalg.norm(predictions[0] - predictions[1])
        errors = np.linalg.norm(predictions - actual, axis=1)

        print(f"\nShoulder boundary check: theta2 = {theta2:.6f} rad")
        print("  Analytical positions (-pi, +pi):", actual)
        print("  Predicted positions (-pi, +pi):", predictions)
        print(f"  Prediction boundary gap: {gap:.6f}")
        print(f"  Error at -pi: {errors[0]:.6f}")
        print(f"  Error at +pi: {errors[1]:.6f}")


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the original raw-angle training visualization",
    )
    args = parser.parse_args()
    if args.train:
        training_demo()
        return
    model_path = Path(__file__).parent / "results" / "periodic_model.npz"
    if not model_path.exists():
        raise SystemExit("Run python experiment.py first to train and save the model.")
    with np.load(model_path, allow_pickle=False) as saved:
        parameters = tuple(
            saved[name].copy()
            for name in ("weights1", "biases1", "weights2", "biases2")
        )
        lengths = tuple(saved["lengths"])
    visualization.live_model(arm.Arm((0.0, 0.5), lengths), parameters)


if __name__ == "__main__":
    main()
