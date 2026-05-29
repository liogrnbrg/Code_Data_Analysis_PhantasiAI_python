import numpy as np
import matplotlib.pyplot as plt

from preprocessing.trial_profiles import build_trial_to_next_event_profiles
from utils.colors import get_subject_color, make_isi_colors
from utils.style import pretty_axes, save_pretty_fig, get_robust_ylims
from utils.plotting_helpers import plot_mean_sd_band
from utils.config import get_config

config = get_config()

# Take the font family setting from the config to ensure consistency across all plots.
#plt.rcParams.update({'font.family': config["plot"]["font"]["family"]})

def plot_activation_profiles_grid_by_participant_and_isi(
    signal_data,
    timing_data,
    participants,
    signal_var,
    y_label,
    fig_title,
    fig_name,
    plots_dir,
    subject_colors,
    config,
):
    """
    Grid plot:
        rows    = participants
        columns = ISIs

    Each panel shows mean ± SD profile for one participant and one ISI.

    Style:
        - participant name shown once per row, separate from shared y-label
        - shared y-label and x-label
        - tick values visible
        - bold figure title and participant row labels
    """

    print(f"  Preparing activation profile plot for: {signal_var}")

    # ---------- Collect ISIs across all included participants ----------
    all_isi = []

    for participant_id in participants:
        timing_p = timing_data[timing_data["participant_id"] == participant_id]
        isi_col = "isi_bin" if "isi_bin" in timing_p.columns else "isi"
        all_isi.extend(timing_p[isi_col].dropna().unique())

    unique_isi = np.array(sorted(set(all_isi)), dtype=float)

    n_participants = len(participants)
    n_isi = len(unique_isi)

    if n_participants == 0 or n_isi == 0:
        print("  No participants or ISIs available for activation profile plot.")
        return

    # ---------- Precompute everything once ----------
    profiles_cache = {}
    all_y = []

    for participant_id in participants:
        print(f"    Computing profiles for {participant_id}...")

        signal_p = signal_data[
            signal_data["participant_id"] == participant_id
        ].sort_values("timestamp")

        timing_p = timing_data[
            timing_data["participant_id"] == participant_id
        ].sort_values("event")

        for isi in unique_isi:
            t_grid, profiles, trial_nums = build_trial_to_next_event_profiles(
                signal_df=signal_p,
                timing_df=timing_p,
                signal_var=signal_var,
                target_isi=isi,
                config=config,
            )

            profiles_cache[(participant_id, isi)] = {
                "t_grid": t_grid,
                "profiles": profiles,
                "trial_nums": trial_nums,
            }

            if profiles.size > 0 and np.any(np.isfinite(profiles)):
                all_y.append(profiles[np.isfinite(profiles)])

            n_valid = profiles.shape[0] if profiles.size > 0 else 0
            print(f"      ISI {isi:.2f}: {n_valid} valid trials")

    if len(all_y) > 0:
        all_y = np.concatenate(all_y)
        ylims = get_robust_ylims(all_y)
    else:
        ylims = (-1, 1)

    # ---------- Figure ----------
    fig_width = max(12, 3.3 * n_isi)
    fig_height = max(4.8, 2.45 * n_participants)

    fig, axes = plt.subplots(
        nrows=n_participants,
        ncols=n_isi,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharey=True,
        sharex=False,
    )

    for row_idx, participant_id in enumerate(participants):
        base_color = get_subject_color(participant_id, subject_colors)
        isi_colors = make_isi_colors(base_color, n_isi)

        for col_idx, isi in enumerate(unique_isi):
            ax = axes[row_idx, col_idx]

            cached = profiles_cache[(participant_id, isi)]
            t_grid = cached["t_grid"]
            profiles = cached["profiles"]

            # ---------- Plot ----------
            if profiles.size == 0 or not np.any(np.isfinite(profiles)):
                ax.text(
                    0.5,
                    0.5,
                    "No valid trials",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                n_valid = 0
            else:
                plotted = plot_mean_sd_band(
                    ax=ax,
                    x=t_grid,
                    profiles=profiles,
                    color=isi_colors[col_idx],
                    face_alpha=config["emg_patterns"]["plot"]["face_alpha"],
                    line_width=config["emg_patterns"]["plot"]["line_width"],
                )

                n_valid = profiles.shape[0]

                if not plotted:
                    ax.text(
                        0.5,
                        0.5,
                        "No valid data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                    )

            # ---------- Column titles only on first row ----------
            if row_idx == 0:
                ax.set_title(
                    f"ISI {isi:.2f} (n={n_valid})",
                    fontsize=13,
                    fontweight="bold",
                    pad=10,
                )
            else:
                ax.set_title("")

            # ---------- Axes ----------
            ax.set_ylim(*ylims)

            if t_grid.size > 0:
                ax.set_xlim(np.nanmin(t_grid), np.nanmax(t_grid))

            # Keep numeric tick labels visible
            ax.tick_params(
                axis="both",
                which="both",
                labelbottom=True,
                labelleft=True,
            )

            pretty_axes(ax)

    # ---------- Shared labels ----------
    if config["emg_patterns"]["profile"]["time_mode"] == "seconds":
        x_label = "Time from current event onset (s)"
    else:
        x_label = "Normalized trial time: current event to next event"

    # Layout first
    fig.subplots_adjust(
        left=0.11,
        right=0.99,
        bottom=0.12,
        top=0.86,
        wspace=0.25,
        hspace=0.25,
    )

    # Shared labels
    fig.supxlabel(x_label, fontsize=14, y=0.04)
    fig.supylabel(y_label, fontsize=14, x=0.03)

    # Main title in bold
    fig.suptitle(fig_title, fontsize=16, fontweight="bold", y=0.97)

    # ---------- Participant row labels (separate from y-label) ----------
    # Use figure coordinates so they are visually distinct from the shared y-label
    for row_idx, participant_id in enumerate(participants):
        ax_left = axes[row_idx, 0]
        bbox = ax_left.get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)

        fig.text(
            0.065,              # horizontal position of participant names
            y_center,
            str(participant_id),
            rotation=90,
            va="center",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )

    save_pretty_fig(fig, fig_name, plots_dir)

    print(f"  Saved {fig_name}")