import numpy as np
import matplotlib.pyplot as plt

from utils.colors import get_subject_color
from utils.paths import make_safe_filename, make_display_label
from utils.style import pretty_axes, save_pretty_fig, get_robust_ylims
from utils.config import get_config

config = get_config()
#plt.rcParams.update({'font.family': config["plot"]["font"]["family"]})

def plot_emg_segment_by_participant(
    segment_df,
    timing_df,
    participant_id,
    plots_dir,
    subject_colors,
    signal_var="emg_processed",
    y_label="Rectified EMG amplitude (µV)",
    fig_suffix="processed_EMG_segment",
):
    """
    Plot one EMG segment for one participant.
    Adds vertical lines for trial event and detected peak_time.
    """

    participant_id = str(participant_id)

    time = segment_df["timestamp"].to_numpy()
    emg = segment_df[signal_var].to_numpy()

    color = get_subject_color(participant_id, subject_colors)

    t_start = np.nanmin(time)
    t_end = np.nanmax(time)

    timing_in_segment = timing_df[
        ((timing_df["event"] >= t_start) & (timing_df["event"] <= t_end))
        | ((timing_df["peak_time"] >= t_start) & (timing_df["peak_time"] <= t_end))
    ].copy()

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(
        time,
        emg,
        color=color,
        linewidth=1.1,
        label="Processed EMG",
    )

    event_color = np.array([0.15, 0.15, 0.15])
    peak_color = np.array([0.55, 0.00, 0.55])

    event_times = timing_in_segment["event"].dropna().unique()
    peak_times = timing_in_segment["peak_time"].dropna().unique()

    for event_time in event_times:
        ax.axvline(
            event_time,
            linestyle="--",
            color=event_color,
            linewidth=1.2,
            alpha=0.9,
        )

    for peak_time in peak_times:
        ax.axvline(
            peak_time,
            linestyle=":",
            color=peak_color,
            linewidth=1.4,
            alpha=0.9,
        )

    # Dummy legend handles
    ax.plot([], [], "--", color=event_color, linewidth=1.2, label="Trial start (event)")
    ax.plot([], [], ":", color=peak_color, linewidth=1.4, label="Detected peak")

    ax.set_xlim(t_start, t_end)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{participant_id} | {make_display_label(fig_suffix)}")

    ax.legend(loc="best", frameon=False)

    pretty_axes(ax)

    filename = f"{make_safe_filename(participant_id)}_{make_safe_filename(fig_suffix)}.png"
    save_pretty_fig(fig, filename, plots_dir)


def plot_accel_segment_by_participant(
    segment_df,
    participant_id,
    plots_dir,
    subject_colors,
    fig_suffix="acceleration_segment",
):
    """
    Plot accel_x, accel_y, accel_z for one participant.
    Same x-axis and same y-axis scale for the 3 acceleration dimensions.
    """

    participant_id = str(participant_id)

    time = segment_df["timestamp"].to_numpy()
    color = get_subject_color(participant_id, subject_colors)

    accel_vars = ["accel_x", "accel_y", "accel_z"]
    accel_labels = ["Acceleration X", "Acceleration Y", "Acceleration Z"]

    all_accel = segment_df[accel_vars].to_numpy().ravel()
    ylims = get_robust_ylims(all_accel, lower_pct=1, upper_pct=99)

    t_start = np.nanmin(time)
    t_end = np.nanmax(time)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(11, 6.5),
        sharex=True,
        sharey=True,
    )

    for ax, var_name, label in zip(axes, accel_vars, accel_labels):
        ax.plot(
            time,
            segment_df[var_name].to_numpy(),
            color=color,
            linewidth=1.1,
        )

        ax.set_title(label)
        ax.set_xlim(t_start, t_end)
        ax.set_ylim(*ylims)

        pretty_axes(ax)

    axes[-1].set_xlabel("Time (s)")
    fig.supylabel("Acceleration")
    fig.suptitle(f"{participant_id} | Acceleration segment")

    filename = f"{make_safe_filename(participant_id)}_{make_safe_filename(fig_suffix)}.png"
    save_pretty_fig(fig, filename, plots_dir)