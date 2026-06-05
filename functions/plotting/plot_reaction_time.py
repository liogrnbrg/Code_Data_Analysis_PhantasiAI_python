import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from utils.colors import get_subject_color
from utils.style import pretty_axes, save_pretty_fig, get_robust_ylims


def plot_reaction_time_one_figure_per_participant(
    rt_data,
    plots_dir,
    subject_colors,
    y_col="reaction_time_norm_delta",
    y_label="Reaction time change from first 10 trials (ms)",
    block_size=20,
    fig_prefix="reaction_time_by_isi_over_trials",
):
    """
    One figure per participant, one subplot per ISI.
    X = trial number within ISI.
    Y = reaction time metric.
    """

    rt_data = rt_data.copy()

    # Safety: create trial_within_isi if missing
    if "trial_within_isi" not in rt_data.columns:
        rt_data = rt_data.sort_values(["participant_id", "isi_bin", "trial_num"])
        rt_data["trial_within_isi"] = (
            rt_data.groupby(["participant_id", "isi_bin"]).cumcount() + 1
        )

    participants = rt_data["participant_id"].dropna().unique()
    isi_values = np.array(sorted(rt_data["isi_bin"].dropna().unique()), dtype=float)

    for participant_id in participants:

        df_participant = rt_data[
            rt_data["participant_id"] == participant_id
        ].copy()

        if df_participant.empty:
            continue

        color = get_subject_color(participant_id, subject_colors)

        valid_y = df_participant.loc[
            df_participant["reaction_time_valid"] & np.isfinite(df_participant[y_col]),
            y_col,
        ]

        if len(valid_y) > 0:
            ylims = get_robust_ylims(valid_y)
        else:
            ylims = (-1, 1)

        invalid_y = ylims[0] + 0.05 * (ylims[1] - ylims[0])

        fig, axes = plt.subplots(
            nrows=len(isi_values),
            ncols=1,
            figsize=(10, 2.8 * len(isi_values)),
            sharex=True,
            sharey=True,
        )

        if len(isi_values) == 1:
            axes = [axes]

        for ax, isi in zip(axes, isi_values):

            df_isi = df_participant[
                df_participant["isi_bin"] == isi
            ].sort_values("trial_within_isi")

            if df_isi.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"ISI {isi:.2f} s")
                pretty_axes(ax)
                continue

            valid = df_isi["reaction_time_valid"] & np.isfinite(df_isi[y_col])
            invalid = ~df_isi["reaction_time_valid"]

            ax.scatter(
                df_isi.loc[valid, "trial_within_isi"],
                df_isi.loc[valid, y_col],
                color=color,
                edgecolor="black",
                linewidth=0.25,
                s=28,
                alpha=0.75,
            )

            if invalid.any():
                ax.scatter(
                    df_isi.loc[invalid, "trial_within_isi"],
                    np.full(invalid.sum(), invalid_y),
                    marker="x",
                    color="red",
                    s=36,
                    alpha=0.9,
                )

            x = df_isi.loc[valid, "trial_within_isi"].to_numpy(dtype=float)
            y = df_isi.loc[valid, y_col].to_numpy(dtype=float)

            title_text = f"ISI {isi:.2f} s"

            if len(x) >= 3:
                res = linregress(x, y)
                x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                y_fit = res.intercept + res.slope * x_fit

                ax.plot(
                    x_fit,
                    y_fit,
                    color=color,
                    linewidth=2.3,
                )

                title_text += (
                    f" | slope = {res.slope:.3f}, "
                    f"r = {res.rvalue:.3f}, "
                    f"p = {res.pvalue:.3g}"
                )

            if "norm" in y_col:
                ax.axhline(0, color="black", linestyle="--", linewidth=1)

            if block_size is not None:
                max_trial = int(np.nanmax(df_isi["trial_within_isi"]))
                for block_start in range(block_size + 1, max_trial + 1, block_size):
                    ax.axvline(
                        block_start - 0.5,
                        color="gray",
                        linestyle=":",
                        linewidth=1,
                        alpha=0.7,
                    )

            ax.set_xlim(0.5, 100.5)
            ax.set_ylim(*ylims)
            ax.set_title(title_text, fontweight="bold")
            pretty_axes(ax)

        axes[-1].set_xlabel("Trial number within ISI")
        fig.supylabel(y_label)

        fig.suptitle(
            f"{participant_id} — EMG reaction time over trials by ISI",
            fontweight="bold",
            y=1.01,
        )

        fig_name = f"{fig_prefix}_{participant_id}.png"

        save_pretty_fig(
            fig,
            fig_name,
            plots_dir,
        )

def plot_reaction_time_variability_one_figure_per_participant(
    rt_data,
    plots_dir,
    subject_colors,
    rt_col="reaction_time_ms",
    rolling_window=10,
    min_periods=5,
    block_size=20,
    fig_prefix="reaction_time_variability_by_isi_over_trials",
):
    """
    Plot rolling reaction time variability over trials.

    One figure per participant:
        - one subplot per ISI
        - x-axis = trial number within ISI
        - y-axis = rolling SD of reaction time

    Variability is computed as the rolling standard deviation of valid RTs
    within each participant/ISI.
    """

    rt_data = rt_data.copy()

    if "trial_within_isi" not in rt_data.columns:
        rt_data = rt_data.sort_values(["participant_id", "isi_bin", "trial_num"])
        rt_data["trial_within_isi"] = (
            rt_data.groupby(["participant_id", "isi_bin"]).cumcount() + 1
        )

    participants = rt_data["participant_id"].dropna().unique()
    isi_values = np.array(sorted(rt_data["isi_bin"].dropna().unique()), dtype=float)

    # Compute rolling variability
    rt_data["reaction_time_rolling_sd"] = np.nan

    for _, idx in rt_data.groupby(["participant_id", "isi_bin"]).groups.items():
        df_group = rt_data.loc[idx].sort_values("trial_within_isi").copy()

        y = df_group[rt_col].where(df_group["reaction_time_valid"])

        rolling_sd = (
            y.rolling(
                window=rolling_window,
                min_periods=min_periods,
                center=True,
            )
            .std()
        )

        rt_data.loc[df_group.index, "reaction_time_rolling_sd"] = rolling_sd

    for participant_id in participants:

        df_participant = rt_data[
            rt_data["participant_id"] == participant_id
        ].copy()

        if df_participant.empty:
            continue

        color = get_subject_color(participant_id, subject_colors)

        valid_y = df_participant["reaction_time_rolling_sd"]
        valid_y = valid_y[np.isfinite(valid_y)]

        if len(valid_y) > 0:
            ylims = get_robust_ylims(valid_y)
        else:
            ylims = (0, 1)

        fig, axes = plt.subplots(
            nrows=len(isi_values),
            ncols=1,
            figsize=(10, 2.8 * len(isi_values)),
            sharex=True,
            sharey=True,
        )

        if len(isi_values) == 1:
            axes = [axes]

        for ax, isi in zip(axes, isi_values):

            df_isi = df_participant[
                df_participant["isi_bin"] == isi
            ].sort_values("trial_within_isi")

            if df_isi.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"ISI {isi:.2f} s")
                pretty_axes(ax)
                continue

            valid = np.isfinite(df_isi["reaction_time_rolling_sd"])

            ax.plot(
                df_isi.loc[valid, "trial_within_isi"],
                df_isi.loc[valid, "reaction_time_rolling_sd"],
                color=color,
                linewidth=2.4,
            )

            ax.scatter(
                df_isi.loc[valid, "trial_within_isi"],
                df_isi.loc[valid, "reaction_time_rolling_sd"],
                color=color,
                edgecolor="black",
                linewidth=0.25,
                s=22,
                alpha=0.65,
            )

            # Mark invalid trials at bottom
            invalid = ~df_isi["reaction_time_valid"]
            if invalid.any():
                invalid_y = ylims[0] + 0.05 * (ylims[1] - ylims[0])

                ax.scatter(
                    df_isi.loc[invalid, "trial_within_isi"],
                    np.full(invalid.sum(), invalid_y),
                    marker="x",
                    color="red",
                    s=36,
                    alpha=0.9,
                )

            # Optional trend line on rolling SD
            x = df_isi.loc[valid, "trial_within_isi"].to_numpy(dtype=float)
            y = df_isi.loc[valid, "reaction_time_rolling_sd"].to_numpy(dtype=float)

            title_text = f"ISI {isi:.2f} s"

            if len(x) >= 3:
                res = linregress(x, y)
                x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                y_fit = res.intercept + res.slope * x_fit

                ax.plot(
                    x_fit,
                    y_fit,
                    color="black",
                    linewidth=1.6,
                    linestyle="--",
                    alpha=0.8,
                )

                title_text += (
                    f" | slope = {res.slope:.3f}, "
                    f"r = {res.rvalue:.3f}, "
                    f"p = {res.pvalue:.3g}"
                )

            # Block boundaries
            if block_size is not None:
                max_trial = int(np.nanmax(df_isi["trial_within_isi"]))

                for block_start in range(block_size + 1, max_trial + 1, block_size):
                    ax.axvline(
                        block_start - 0.5,
                        color="gray",
                        linestyle=":",
                        linewidth=1,
                        alpha=0.7,
                    )

            ax.set_xlim(0.5, 100.5)
            ax.set_ylim(*ylims)
            ax.set_title(title_text, fontweight="bold")
            pretty_axes(ax)

        axes[-1].set_xlabel("Trial number within ISI")
        fig.supylabel(f"Rolling RT variability, SD over {rolling_window} trials (ms)")

        fig.suptitle(
            f"{participant_id} — reaction time variability over trials by ISI",
            fontweight="bold",
            y=1.01,
        )

        fig_name = f"{fig_prefix}_{participant_id}.png"

        save_pretty_fig(
            fig,
            fig_name,
            plots_dir,
        )