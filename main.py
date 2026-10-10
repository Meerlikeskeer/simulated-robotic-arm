import datasets
import visualization
import arm
import neural_network


def main() -> None:
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

    visualization.plot_test_errors(test_dataset, parameters)


if __name__ == "__main__":
    main()
