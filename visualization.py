import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

import arm
import kinematics
import neural_network


def plot_test_errors(dataset, parameters):
    """Color unseen angle pairs by their Euclidean tip-position error."""
    inputs = dataset[:, :2]
    targets = dataset[:, 2:]
    predictions = neural_network.predict_positions(inputs, parameters)
    errors = np.linalg.norm(predictions - targets, axis=1)

    fig, ax = plt.subplots(figsize=(12, 10))
    points = ax.scatter(
        inputs[:, 0], inputs[:, 1], c=errors, cmap="viridis", s=20, vmin=0
    )
    fig.colorbar(points, ax=ax, label="Positional error (link-length units)")
    worst = np.argmax(errors)
    ax.scatter(
        inputs[worst, 0],
        inputs[worst, 1],
        s=350,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        zorder=5,
        label="Largest sampled error",
    )
    ax.set_title("Unseen test error by joint configuration")
    ax.set_xlabel("θ1 (rad)")
    ax.set_ylabel("θ2 (rad)")
    padding = 0.2
    ax.set_xlim(-math.pi - padding, math.pi + padding)
    ax.set_ylim(-math.pi - padding, math.pi + padding)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    plt.show()


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


def watch_training(inputs, targets, learning_rate=0.01, epochs=1000):
    import numpy as np

    import neural_network

    fig, (loss_ax, position_ax) = plt.subplots(1, 2, figsize=(11, 5))

    loss_ax.set_title("Training loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Mean half-squared error")
    loss_ax.set_yscale("log")
    loss_ax.grid(True)
    (loss_line,) = loss_ax.plot([], [])

    position_ax.set_title("Targets and current predictions")
    position_ax.scatter(
        targets[:, 0],
        targets[:, 1],
        s=15,
        alpha=0.4,
        label="Targets",
    )
    predictions_plot = position_ax.scatter(
        [],
        [],
        s=15,
        color="orange",
        alpha=0.6,
        label="Predictions",
    )
    position_ax.set_xlabel("x")
    position_ax.set_ylabel("y")
    position_ax.set_aspect("equal")
    position_ax.grid(True)
    position_ax.legend()

    limit = max(1.0, float(np.abs(targets).max()) * 1.2)
    position_ax.set_xlim(-limit, limit)
    position_ax.set_ylim(-limit, limit)

    epoch_history = []
    loss_history = []

    def update(epoch, weights1, biases1, weights2, biases2):
        # Each row is an example, hence the transposed weight matrices.
        hidden = np.tanh(inputs @ weights1.T + biases1)
        predictions = hidden @ weights2.T + biases2

        errors = predictions - targets
        loss = 0.5 * np.mean(np.sum(errors**2, axis=1))

        epoch_history.append(epoch)
        loss_history.append(max(float(loss), 1e-15))
        loss_line.set_data(epoch_history, loss_history)
        loss_ax.relim()
        loss_ax.autoscale_view()

        predictions_plot.set_offsets(predictions)
        position_ax.set_title(f"Predictions after epoch {epoch}")

        fig.canvas.draw_idle()
        plt.pause(0.001)

    plt.show(block=False)
    parameters = neural_network.train_nonlinear_network(
        inputs,
        targets,
        learning_rate,
        epochs,
        callback=update,
    )
    plt.show()
    return parameters


def live_model(arm1, parameters, show=True):
    """Compare the analytical arm with a learned tip; no physical control."""
    positions = kinematics.joint_positions(arm1)
    ax, line = plot_arm(positions, *arm1.lengths)
    fig = ax.figure
    fig.subplots_adjust(bottom=0.28)
    line.set_label("Analytical arm")
    predicted = ax.scatter(
        [], [], marker="x", s=120, color="orange", label="Learned tip", zorder=4
    )
    (error_line,) = ax.plot([], [], "r--", label="Position error")
    text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")
    ax.legend(loc="lower left")
    sliders = [
        Slider(
            fig.add_axes([0.2, y, 0.65, 0.03]),
            f"θ{i + 1} (rad)",
            -math.pi,
            math.pi,
            valinit=arm1.angles[i],
        )
        for i, y in enumerate((0.12, 0.06))
    ]

    def update(_):
        angles = np.array([s.val for s in sliders])
        actual = kinematics.joint_positions(arm.Arm(tuple(angles), arm1.lengths))
        tip = neural_network.predict_positions(angles[None, :], parameters)[0]
        line.set_data(actual[:, 0], actual[:, 1])
        predicted.set_offsets(tip[None, :])
        error_line.set_data([actual[-1, 0], tip[0]], [actual[-1, 1], tip[1]])
        text.set_text(f"Positional error: {np.linalg.norm(tip - actual[-1]):.5f}")
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)
    update(None)
    # Retain controls even when used non-blocking.
    fig._arm_sliders = sliders
    if show:
        plt.show()
    return fig, sliders
