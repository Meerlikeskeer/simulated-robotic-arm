from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    """record of a robotic arm's joint angles and link lengths."""

    angles: tuple[float, ...]
    lengths: tuple[float, ...]
