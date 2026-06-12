
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import wilcoxon

FS = 24

plt.rcParams.update({
    "font.family": "serif",
    "font.size": FS,
    "mathtext.fontset": "cm",
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


def remove_top_right(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_metric_array(data, preferred_name, fallback_name=None):
    if preferred_name in data:
        return data[preferred_name]

    if fallback_name is not None and fallback_name in data:
        return data[fallback_name]

    raise KeyError(f"Could not find {preferred_name} in metrics file.")


def load_metric_scalar(data, preferred_name, fallback_name=None, default=None):
    if preferred_name in data:
        return float(data[preferred_name])

    if fallback_name is not None and fallback_name in data:
        return float(data[fallback_name])

    if default is not None:
        return default

    raise KeyError(f"Could not find {preferred_name} in metrics file.")


def print_metrics(metrics_file, label):
    data = np.load(metrics_file, allow_pickle=True)

    success = data["success"]
    gate_steps = data["gate_steps"]

    key_visible_times = load_metric_array(
        data,
        "key_visible_times",
        fallback_name="key_visible_time",
    )

    pickup_times = load_metric_array(
        data,
        "pickup_times",
        fallback_name=None,
    ) if "pickup_times" in data else None

    door_visible_times = load_metric_array(
        data,
        "door_visible_times",
        fallback_name="door_visible_time",
    )

    '''door_access_times = load_metric_array(
        data,
        "door_access_times",
        fallback_name="door_access_time",
    ) if ("door_access_times" in data or "door_access_time" in data) else None'''

    success_rate = float(data["success_rate"])

    mean_gate_steps = float(data["mean_gate_steps"])
    std_gate_steps = float(data["std_gate_steps"])

    mean_key_visible = load_metric_scalar(data, "mean_key_visible")
    std_key_visible = load_metric_scalar(data, "std_key_visible")

    mean_pickup = load_metric_scalar(
        data,
        "mean_pickup",
        fallback_name="mean_pickup_time",
        default=None,
    ) if ("mean_pickup" in data or "mean_pickup_time" in data) else None

    std_pickup = load_metric_scalar(
        data,
        "std_pickup",
        fallback_name="std_pickup_time",
        default=None,
    ) if ("std_pickup" in data or "std_pickup_time" in data) else None

    mean_door_visible = load_metric_scalar(data, "mean_door_visible")
    std_door_visible = load_metric_scalar(data, "std_door_visible")

    mean_door_access = load_metric_scalar(
        data,
        "mean_door_access",
        fallback_name="mean_door_access_time",
        default=None,
    ) if ("mean_door_access" in data or "mean_door_access_time" in data) else None

    std_door_access = load_metric_scalar(
        data,
        "std_door_access",
        fallback_name="std_door_access_time",
        default=None,
    ) if ("std_door_access" in data or "std_door_access_time" in data) else None

    print("\n" + label)
    print("-" * len(label))
    print(f"Number of episodes: {len(success)}")
    print(f"Success rate: {100.0 * success_rate:.1f}%")
    print(f"Mean steps: {mean_gate_steps:.2f} +/- {std_gate_steps:.2f}")
    print(f"First key sensed: {mean_key_visible:.2f} +/- {std_key_visible:.2f}")

    if mean_pickup is not None:
        print(f"Key pickup: {mean_pickup:.2f} +/- {std_pickup:.2f}")

    print(f"First door sensed: {mean_door_visible:.2f} +/- {std_door_visible:.2f}")

    if mean_pickup is not None:
        print(f"Mean sensed-to-pickup delay: {mean_pickup - mean_key_visible:.2f}")


def plot_trajectory(data_file, output_file):
    data = np.load(data_file, allow_pickle=True)

    trajectory = data["trajectory"]
    key_candidates = data["key_candidates"]
    door_candidates = data["door_candidates"]
    true_key = data["true_key"]
    true_door = data["true_door"]

    pickup_radius = float(data["pickup_radius"])
    door_radius = float(data["door_radius"])

    x_min = float(data["x_min"])
    x_max = float(data["x_max"])
    y_min = float(data["y_min"])
    y_max = float(data["y_max"])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(trajectory[:, 0], trajectory[:, 1], linewidth=2)

    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        marker="x",
        color="red",
        linewidth=3,
        s=120,
        label="start",
    )

    for p in key_candidates:
        ax.add_patch(
            plt.Circle(
                p,
                pickup_radius,
                fill=False,
                edgecolor="orange",
                linewidth=2,
                zorder=-2,
            )
        )

    for p in door_candidates:
        ax.add_patch(
            plt.Circle(
                p,
                door_radius,
                fill=False,
                edgecolor="purple",
                linewidth=2,
                zorder=-2,
            )
        )

    ax.scatter(
        true_key[0],
        true_key[1],
        marker="*",
        color="orange",
        edgecolor="black",
        s=400,
        label="true key",
    )

    ax.scatter(
        true_door[0],
        true_door[1],
        marker="D",
        color="purple",
        edgecolor="black",
        s=200,
        label="true door",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_weights(data_file, output_file):
    data = np.load(data_file, allow_pickle=True)

    weights = data["weights"]
    dt = float(data["robotarium_dt"])
    primitive_names = data["primitive_names"]

    time = np.arange(weights.shape[0]) * dt

    fig, ax = plt.subplots(figsize=(8, 5))

    weight_colors = [
        "tab:green",
        "tab:pink",
        "tab:cyan",
    ]

    for i, name in enumerate(primitive_names):
        ax.plot(
            time,
            weights[:, i],
            linewidth=2,
            label=str(name),
            color=weight_colors[i % len(weight_colors)],
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Weights")
    ax.set_xlim(0, time[-1])
    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    remove_top_right(ax)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_entropy(data_file, output_file):
    data = np.load(data_file, allow_pickle=True)

    key_entropy = data["key_entropy"]
    door_entropy = data["door_entropy"]
    dt = float(data["robotarium_dt"])

    time = np.arange(len(key_entropy)) * dt

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        time,
        key_entropy,
        linewidth=2,
        label=r"$H(b^{\mathrm{K}})$",
        color="tab:orange",
    )

    ax.plot(
        time,
        door_entropy,
        linewidth=2,
        label=r"$H(b^{\mathrm{D}})$",
        color="tab:purple",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Belief entropy")
    ax.set_xlim(0, time[-1])

    # For NUM_CANDIDATES = 6, max entropy is log(6)
    if "key_candidates" in data:
        n_candidates = len(data["key_candidates"])
        ax.set_ylim(-0.02, np.log(n_candidates) + 0.05)

    #ax.legend(frameon=False)
    remove_top_right(ax)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_omega(data_file, output_file):
    data = np.load(data_file, allow_pickle=True)

    omega = data["omega"]
    dt = float(data["robotarium_dt"])
    time = np.arange(len(omega)) * dt

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(time, omega, linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\omega$ [rad/s]")
    ax.set_xlim(0, time[-1])

    remove_top_right(ax)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_p_values(epi_metrics_file, no_epi_metrics_file):

    epi = np.load(epi_metrics_file, allow_pickle=True)
    no_epi = np.load(no_epi_metrics_file, allow_pickle=True)

    tests = [
        ("door_access_times", "Door access"),
        ("key_visible_times", "Key visible"),
        ("pickup_times", "Key pickup"),
        ("door_visible_times", "Door visible"),
    ]

    print("\nPaired Wilcoxon p-values")
    print("------------------------")

    for key, label in tests:
        if key not in epi or key not in no_epi:
            print(f"{label}: skipped, missing '{key}'")
            continue

        x = np.asarray(epi[key], dtype=float)
        y = np.asarray(no_epi[key], dtype=float)

        if len(x) != len(y):
            print(f"{label}: skipped, different number of episodes")
            continue

        diff = x - y

        if np.allclose(diff, 0.0):
            print(f"{label}: p = 1.0  (all paired differences are zero)")
            continue

        stat, p = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox"
        )

        print(f"{label}: p = {p:.4g}")


if __name__ == "__main__":
    # Print aggregate metrics over all episodes.
    epi_metrics_file = os.path.join(
        RESULTS_DIR,
        "robotarium_epi_metrics.npz",
    )

    no_epi_metrics_file = os.path.join(
        RESULTS_DIR,
        "robotarium_no_epi_metrics.npz",
    )

    print_metrics(
        epi_metrics_file,
        "With epistemic term",
    )

    print_metrics(
        no_epi_metrics_file,
        "Without epistemic term",
    )

    print_p_values(
        epi_metrics_file,
        no_epi_metrics_file,
    )

    # Plot only the first episode.
    epi_first_episode = os.path.join(
        RESULTS_DIR,
        "robotarium_epi_ep000_data.npz",
    )
    no_epi_first_episode = os.path.join(
        RESULTS_DIR,
        "robotarium_no_epi_ep000_data.npz",
    )

    plot_trajectory(
        epi_first_episode,
        os.path.join(FIGURES_DIR, "robotarium_epi_trajectory.pdf"),
    )
    plot_weights(
        epi_first_episode,
        os.path.join(FIGURES_DIR, "robotarium_epi_weights.pdf"),
    )
    plot_omega(
        epi_first_episode,
        os.path.join(FIGURES_DIR, "robotarium_epi_omega.pdf"),
    )
    plot_entropy(
        epi_first_episode,
        os.path.join(FIGURES_DIR, "robotarium_epi_entropy.pdf"),
    )

    plot_trajectory(
        no_epi_first_episode,
        os.path.join(FIGURES_DIR, "robotarium_no_epi_trajectory.pdf"),
    )
    plot_weights(
        no_epi_first_episode,
        os.path.join(FIGURES_DIR, "robotarium_no_epi_weights.pdf"),
    )
    plot_omega(
        no_epi_first_episode,
        os.path.join(FIGURES_DIR, "robotarium_no_epi_omega.pdf"),
    )