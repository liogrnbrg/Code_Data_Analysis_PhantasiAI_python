from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from utils.colors import get_subject_color
from utils.style import pretty_axes

def animate_trial_trajectory_2d(
    signal_data,
    timing_data,
    participant_id,
    trial_num,
    plots_dir,
    subject_colors,
    x_var="position_x",
    y_var="position_y",
    fps=20,
    n_frames=120,
    trail=True,
    filename=None,
):
    """
    Create a 2D GIF of one trial trajectory.

    The trajectory uses proxy position variables derived from trial-wise
    double integration of acceleration.

    Example:
        x_var = "position_x"
        y_var = "position_y"

    Important:
        This is a qualitative movement proxy, not validated spatial kinematics.
    """

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    timing_p = timing_data[
        timing_data["participant_id"] == participant_id
    ].sort_values("event").reset_index(drop=True)

    signal_p = signal_data[
        signal_data["participant_id"] == participant_id
    ].sort_values("timestamp").reset_index(drop=True)

    if trial_num not in timing_p["trial_num"].values:
        raise ValueError(f"Trial {trial_num} not found for participant {participant_id}")

    trial_idx = timing_p.index[timing_p["trial_num"] == trial_num][0]

    if trial_idx >= len(timing_p) - 1:
        raise ValueError("Cannot animate last trial because next event is needed.")

    event_start = timing_p.loc[trial_idx, "event"]
    event_end = timing_p.loc[trial_idx + 1, "event"]
    isi = timing_p.loc[trial_idx, "isi"] if "isi" in timing_p.columns else np.nan

    mask = (
        (signal_p["timestamp"] >= event_start)
        & (signal_p["timestamp"] < event_end)
    )

    trial_df = signal_p.loc[mask].copy()

    if trial_df.empty:
        raise ValueError("No signal data found for this trial window.")

    for col in [x_var, y_var]:
        if col not in trial_df.columns:
            raise ValueError(f"{col} not found in signal_data.")

    t = trial_df["timestamp"].to_numpy(dtype=float) - event_start
    x = trial_df[x_var].to_numpy(dtype=float)
    y = trial_df[y_var].to_numpy(dtype=float)

    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t = t[valid]
    x = x[valid]
    y = y[valid]

    if len(t) < 3:
        raise ValueError("Not enough valid samples to animate.")

    # Downsample/interpolate to fixed number of frames
    frame_t = np.linspace(t[0], t[-1], n_frames)
    x_i = np.interp(frame_t, t, x)
    y_i = np.interp(frame_t, t, y)

    color = get_subject_color(participant_id, subject_colors)

    x_pad = 0.10 * (np.nanmax(x_i) - np.nanmin(x_i) + 1e-9)
    y_pad = 0.10 * (np.nanmax(y_i) - np.nanmin(y_i) + 1e-9)

    fig, ax = plt.subplots(figsize=(6, 6))

    if trail:
        line, = ax.plot([], [], color=color, linewidth=2.2)
    else:
        line = None

    point, = ax.plot([], [], "o", color=color, markersize=8)

    ax.scatter(x_i[0], y_i[0], color="black", s=45, label="Start", zorder=3)
    ax.scatter(x_i[-1], y_i[-1], color="gray", s=45, label="End", zorder=3)

    ax.set_xlim(np.nanmin(x_i) - x_pad, np.nanmax(x_i) + x_pad)
    ax.set_ylim(np.nanmin(y_i) - y_pad, np.nanmax(y_i) + y_pad)

    ax.set_xlabel(x_var.replace("_", " "))
    ax.set_ylabel(y_var.replace("_", " "))

    ax.set_title(
        f"{participant_id} | Trial {trial_num} | ISI {isi:.2f} s",
        fontweight="bold",
    )

    ax.legend(frameon=False)
    pretty_axes(ax)

    time_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )

    def init():
        if trail:
            line.set_data([], [])
        point.set_data([], [])
        time_text.set_text("")
        return tuple(obj for obj in [line, point, time_text] if obj is not None)

    def update(frame):
        if trail:
            line.set_data(x_i[: frame + 1], y_i[: frame + 1])

        point.set_data([x_i[frame]], [y_i[frame]])
        time_text.set_text(f"t = {frame_t[frame]:.2f} s")

        return tuple(obj for obj in [line, point, time_text] if obj is not None)

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=1000 / fps,
        blit=True,
    )

    if filename is None:
        filename = (
            f"{participant_id}_trial_{trial_num}_"
            f"{x_var}_vs_{y_var}_trajectory.gif"
        )

    out_path = plots_dir / filename

    anim.save(
        out_path,
        writer=PillowWriter(fps=fps),
    )

    plt.close(fig)

    print(f"Saved GIF: {out_path}")

    return out_path

def animate_trial_trajectory_3d(
    signal_data,
    timing_data,
    participant_id,
    trial_num,
    plots_dir,
    subject_colors,
    x_var="position_x",
    y_var="position_y",
    z_var="position_z",
    fps=20,
    n_frames=120,
    trail=True,
    elev=25,
    azim=45,
    filename=None,
):
    """
    Create a 3D GIF of one trial trajectory.

    The trajectory uses proxy position variables derived from trial-wise
    double integration of acceleration.

    Important:
        This is a qualitative movement proxy, not validated spatial kinematics.
    """

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    timing_p = timing_data[
        timing_data["participant_id"] == participant_id
    ].sort_values("event").reset_index(drop=True)

    signal_p = signal_data[
        signal_data["participant_id"] == participant_id
    ].sort_values("timestamp").reset_index(drop=True)

    if trial_num not in timing_p["trial_num"].values:
        raise ValueError(f"Trial {trial_num} not found for participant {participant_id}")

    trial_idx = timing_p.index[timing_p["trial_num"] == trial_num][0]

    if trial_idx >= len(timing_p) - 1:
        raise ValueError("Cannot animate last trial because next event is needed.")

    event_start = timing_p.loc[trial_idx, "event"]
    event_end = timing_p.loc[trial_idx + 1, "event"]
    isi = timing_p.loc[trial_idx, "isi"] if "isi" in timing_p.columns else np.nan

    mask = (
        (signal_p["timestamp"] >= event_start)
        & (signal_p["timestamp"] < event_end)
    )

    trial_df = signal_p.loc[mask].copy()

    if trial_df.empty:
        raise ValueError("No signal data found for this trial window.")

    for col in [x_var, y_var, z_var]:
        if col not in trial_df.columns:
            raise ValueError(f"{col} not found in signal_data.")

    t = trial_df["timestamp"].to_numpy(dtype=float) - event_start
    x = trial_df[x_var].to_numpy(dtype=float)
    y = trial_df[y_var].to_numpy(dtype=float)
    z = trial_df[z_var].to_numpy(dtype=float)

    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    t = t[valid]
    x = x[valid]
    y = y[valid]
    z = z[valid]

    if len(t) < 3:
        raise ValueError("Not enough valid samples to animate.")

    # Interpolate to a fixed number of frames
    frame_t = np.linspace(t[0], t[-1], n_frames)
    x_i = np.interp(frame_t, t, x)
    y_i = np.interp(frame_t, t, y)
    z_i = np.interp(frame_t, t, z)

    color = get_subject_color(participant_id, subject_colors)

    # Make axis limits roughly equal
    x_min, x_max = np.nanmin(x_i), np.nanmax(x_i)
    y_min, y_max = np.nanmin(y_i), np.nanmax(y_i)
    z_min, z_max = np.nanmin(z_i), np.nanmax(z_i)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    half_range = 0.5 * max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min,
        1e-6,
    )
    half_range *= 1.15

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    if trail:
        line, = ax.plot([], [], [], color=color, linewidth=2.2)
    else:
        line = None

    point, = ax.plot([], [], [], "o", color=color, markersize=7)

    ax.scatter([x_i[0]], [y_i[0]], [z_i[0]], color="black", s=45, label="Start")
    ax.scatter([x_i[-1]], [y_i[-1]], [z_i[-1]], color="gray", s=45, label="End")

    ax.set_xlim(x_mid - half_range, x_mid + half_range)
    ax.set_ylim(y_mid - half_range, y_mid + half_range)
    ax.set_zlim(z_mid - half_range, z_mid + half_range)

    ax.set_xlabel(x_var.replace("_", " "))
    ax.set_ylabel(y_var.replace("_", " "))
    ax.set_zlabel(z_var.replace("_", " "))

    ax.set_title(
        f"{participant_id} | Trial {trial_num} | ISI {isi:.2f} s",
        fontweight="bold",
    )

    ax.legend(frameon=False)

    # Light styling
    ax.grid(True, alpha=0.25)

    time_text = ax.text2D(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )

    def init():
        if trail:
            line.set_data([], [])
            line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        time_text.set_text("")
        artists = [point, time_text]
        if trail:
            artists.insert(0, line)
        return tuple(artists)

    def update(frame):
        if trail:
            line.set_data(x_i[: frame + 1], y_i[: frame + 1])
            line.set_3d_properties(z_i[: frame + 1])

        point.set_data([x_i[frame]], [y_i[frame]])
        point.set_3d_properties([z_i[frame]])
        time_text.set_text(f"t = {frame_t[frame]:.2f} s")

        artists = [point, time_text]
        if trail:
            artists.insert(0, line)
        return tuple(artists)

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=1000 / fps,
        blit=False,  # safer in 3D
    )

    if filename is None:
        filename = (
            f"{participant_id}_trial_{trial_num}_"
            f"{x_var}_vs_{y_var}_vs_{z_var}_trajectory_3d.gif"
        )

    out_path = plots_dir / filename

    anim.save(
        out_path,
        writer=PillowWriter(fps=fps),
    )

    plt.close(fig)

    print(f"Saved 3D GIF: {out_path}")

    return out_path

def animate_trial_sequence_2d(
    signal_data,
    timing_data,
    participant_id,
    trial_nums,
    plots_dir,
    subject_colors,
    x_var="position_x",
    y_var="position_y",
    fps=20,
    frames_per_trial=50,
    pause_frames=8,
    reset_to_origin=True,
    trail=True,
    filename=None,
):
    """
    Animate a sequence of trials in 2D.

    Each trial is shown one after the other in the same axes.
    The previous trial disappears when the new one starts.

    Useful to visually compare movement direction/shape across trials.
    """

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    timing_p = timing_data[
        timing_data["participant_id"] == participant_id
    ].sort_values("event").reset_index(drop=True)

    signal_p = signal_data[
        signal_data["participant_id"] == participant_id
    ].sort_values("timestamp").reset_index(drop=True)

    color = get_subject_color(participant_id, subject_colors)

    trajectories = []

    for trial_num in trial_nums:

        if trial_num not in timing_p["trial_num"].values:
            print(f"Warning: trial {trial_num} not found for {participant_id}. Skipping.")
            continue

        trial_idx = timing_p.index[timing_p["trial_num"] == trial_num][0]

        if trial_idx >= len(timing_p) - 1:
            print(f"Warning: trial {trial_num} is the last trial, skipping.")
            continue

        event_start = timing_p.loc[trial_idx, "event"]
        event_end = timing_p.loc[trial_idx + 1, "event"]
        isi = timing_p.loc[trial_idx, "isi"] if "isi" in timing_p.columns else np.nan

        mask = (
            (signal_p["timestamp"] >= event_start)
            & (signal_p["timestamp"] < event_end)
        )

        trial_df = signal_p.loc[mask].copy()

        if trial_df.empty:
            print(f"Warning: no data for trial {trial_num}. Skipping.")
            continue

        for col in [x_var, y_var]:
            if col not in trial_df.columns:
                raise ValueError(f"{col} not found in signal_data.")

        t = trial_df["timestamp"].to_numpy(dtype=float) - event_start
        x = trial_df[x_var].to_numpy(dtype=float)
        y = trial_df[y_var].to_numpy(dtype=float)

        valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
        t = t[valid]
        x = x[valid]
        y = y[valid]

        if len(t) < 3:
            print(f"Warning: not enough valid samples for trial {trial_num}. Skipping.")
            continue

        frame_t = np.linspace(t[0], t[-1], frames_per_trial)
        x_i = np.interp(frame_t, t, x)
        y_i = np.interp(frame_t, t, y)

        if reset_to_origin:
            x_i = x_i - x_i[0]
            y_i = y_i - y_i[0]

        trajectories.append({
            "trial_num": trial_num,
            "isi": isi,
            "t": frame_t,
            "x": x_i,
            "y": y_i,
        })

    if len(trajectories) == 0:
        raise ValueError("No valid trials found to animate.")

    # ---------- Shared axis limits across all included trials ----------
    all_x = np.concatenate([traj["x"] for traj in trajectories])
    all_y = np.concatenate([traj["y"] for traj in trajectories])

    x_pad = 0.10 * (np.nanmax(all_x) - np.nanmin(all_x) + 1e-9)
    y_pad = 0.10 * (np.nanmax(all_y) - np.nanmin(all_y) + 1e-9)

    # ---------- Figure ----------
    fig, ax = plt.subplots(figsize=(6, 6))

    if trail:
        line, = ax.plot([], [], color=color, linewidth=2.2)
    else:
        line = None

    point, = ax.plot([], [], "o", color=color, markersize=8)

    ax.set_xlim(np.nanmin(all_x) - x_pad, np.nanmax(all_x) + x_pad)
    ax.set_ylim(np.nanmin(all_y) - y_pad, np.nanmax(all_y) + y_pad)

    ax.set_xlabel(x_var.replace("_", " "))
    ax.set_ylabel(y_var.replace("_", " "))

    pretty_axes(ax)

    time_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )

    # One block = one trial animation + short pause
    block_size = frames_per_trial + pause_frames
    total_frames = len(trajectories) * block_size

    def init():
        if trail:
            line.set_data([], [])
        point.set_data([], [])
        time_text.set_text("")
        return tuple(obj for obj in [line, point, time_text] if obj is not None)

    def update(frame):
        trial_block_idx = frame // block_size
        local_frame = frame % block_size

        traj = trajectories[trial_block_idx]

        draw_frame = min(local_frame, frames_per_trial - 1)

        x_now = traj["x"][: draw_frame + 1]
        y_now = traj["y"][: draw_frame + 1]

        if trail:
            line.set_data(x_now, y_now)

        point.set_data([traj["x"][draw_frame]], [traj["y"][draw_frame]])

        ax.set_title(
            f"{participant_id} | Trial {traj['trial_num']} | ISI {traj['isi']:.2f} s",
            fontweight="bold",
        )

        time_text.set_text(f"t = {traj['t'][draw_frame]:.2f} s")

        return tuple(obj for obj in [line, point, time_text] if obj is not None)

    anim = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        init_func=init,
        interval=1000 / fps,
        blit=False,
    )

    if filename is None:
        first_trial = trajectories[0]["trial_num"]
        last_trial = trajectories[-1]["trial_num"]
        filename = (
            f"{participant_id}_trials_{first_trial}_to_{last_trial}_"
            f"{x_var}_vs_{y_var}_sequence.gif"
        )

    out_path = plots_dir / filename

    anim.save(
        out_path,
        writer=PillowWriter(fps=fps),
    )

    plt.close(fig)

    print(f"Saved GIF: {out_path}")

    return out_path