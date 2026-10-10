import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math
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


def interactive_arm(theta1, theta2, l1, l2):
    positions = kinematics.joint_positions(theta1, theta2, l1, l2)
    ax, line = plot_arm(positions, l1, l2)
    fig = ax.figure
    fig.subplots_adjust(bottom=0.25)

    theta1_ax = fig.add_axes([0.2, 0.12, 0.65, 0.03])
    theta2_ax = fig.add_axes([0.2, 0.06, 0.65, 0.03])

    theta1_slider = Slider(theta1_ax, "θ1 (rad)", -math.pi, math.pi, valinit=theta1)
    theta2_slider = Slider(theta2_ax, "θ2 (rad)", -math.pi, math.pi, valinit=theta2)

    def update(value):
        new_positions = kinematics.joint_positions(
            theta1_slider.val, theta2_slider.val, l1, l2
        )
        line.set_data(new_positions[:, 0], new_positions[:, 1])
        fig.canvas.draw_idle()

    theta1_slider.on_changed(update)
    theta2_slider.on_changed(update)

    plt.show()
