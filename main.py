import datasets
import visualization
import arm


def main() -> None:
    # arm1 = arm.Arm(angles=(0.0, 0.5), lengths=(1.0, 1.0))
    # visualization.interactive_arm(arm1)
    arm1 = arm.Arm(angles=(0.0, 0.0), lengths=(2.0, 1.0))
    dataset = datasets.dataset_builder(1000, arm1)
    visualization.plot_datasets(dataset)


if __name__ == "__main__":
    main()
