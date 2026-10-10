import math
import numpy as np

from arm import Arm
from datasets import dataset_builder


def test_dataset_builder():
    arm1 = Arm(angles=(0.0, 0.0), lengths=(1.0, 1.0))
    dataset = dataset_builder(num_samples=10, arm1=arm1)
    assert dataset.shape == (10, 4)
    print("Test passed")
