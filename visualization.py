import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math
import arm
import kinematics


def plot_arm(positions, l1, l2):

    x_coordinates = positions[:, 0]
    y_coordinates = positions[:, 1]

    limit = 1.1 * (l1 + l2)

    fig, ax = plt.subplots()

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    (line,) = ax.plot(x_coordinates, y_coordinates, "o-")
    ax.set_aspect("equal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    return ax, line


def interactive_arm(arm1: arm.Arm):
    positions = kinematics.joint_positions(arm1)
    ax, line = plot_arm(positions, arm1.lengths[0], arm1.lengths[1])
    fig = ax.figure
    fig.subplots_adjust(bottom=0.25)

    theta1_ax = fig.add_axes([0.2, 0.12, 0.65, 0.03])
    theta2_ax = fig.add_axes([0.2, 0.06, 0.65, 0.03])

    theta1_slider = Slider(
        theta1_ax, "θ1 (rad)", -math.pi, math.pi, valinit=arm1.angles[0]
    )
    theta2_slider = Slider(
        theta2_ax, "θ2 (rad)", -math.pi, math.pi, valinit=arm1.angles[1]
    )

    def update(value):
        updated_arm = arm.Arm(
            angles=(theta1_slider.val, theta2_slider.val),
            lengths=arm1.lengths,
        )
        new_positions = kinematics.joint_positions(updated_arm)
        line.set_data(new_positions[:, 0], new_positions[:, 1])
        fig.canvas.draw_idle()

    theta1_slider.on_changed(update)
    theta2_slider.on_changed(update)

    plt.show()


def plot_datasets(dataset):
    fig, (joint_ax, workspace_ax) = plt.subplots(1, 2, figsize=(11, 5))

    joint_ax.scatter(dataset[:, 0], dataset[:, 1], s=8, alpha=0.4)
    joint_ax.set_title("Joint space")
    joint_ax.set_xlabel("theta1 (rad)")
    joint_ax.set_ylabel("theta2 (rad)")
    joint_ax.grid(True)

    workspace_ax.scatter(dataset[:, 2], dataset[:, 3], s=8, alpha=0.4)
    workspace_ax.set_title("Workspace")
    workspace_ax.set_xlabel("Tip x")
    workspace_ax.set_ylabel("Tip y")
    workspace_ax.set_aspect("equal")
    workspace_ax.grid(True)

    fig.tight_layout()
    plt.show()
