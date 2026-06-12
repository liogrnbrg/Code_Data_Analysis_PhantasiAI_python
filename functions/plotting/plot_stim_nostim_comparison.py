from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from utils.colors import get_subject_color
from utils.style import pretty_axes, save_pretty_fig


def plot_stim_nostim_rt_overlay(
    long_data,
    trend_stats,
    plots_dir,
    subject_colors,
    y_col="reaction_time_centered_ms",
    y_label="RT change from first 10 trials (ms)",
    block_size=80,
    xlim=(0.5, 400.5),
    fig_prefix="stim_vs_nostim_rt_overlay",
):
    """
    Plot STIM and NOSTIM reaction-time trajectories.

    One figure per base participant:
        - one subplot per ISI
        - global trial number on the x-axis
        - individual observations as scatter points
        - one linear regression per condition
        - interaction p-value testing whether the two slopes differ

    Colors are taken from config["plot"]["subject_colors"] using the
    original session identifiers, for example:
        Lio_STIM
        Lio_NOSTIM
    """

    long_data = long_data.copy()

    required_cols = {
        "base_participant",
        "participant_id",
        "condition",
        "isi_bin",
        "trial_num",
        "reaction_time_valid",
        y_col,
    }

    missing = required_cols.difference(long_data.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    for participant in long_data[
        "base_participant"
    ].dropna().unique():

        participant_data = long_data[
            long_data["base_participant"] == participant
        ].copy()

        isi_values = sorted(
            participant_data["isi_bin"]
            .dropna()
            .unique()
        )

        if not isi_values:
            continue

        valid_global_y = participant_data.loc[
            participant_data["reaction_time_valid"].astype(bool)
            & np.isfinite(participant_data[y_col]),
            y_col,
        ]

        if len(valid_global_y) > 0:
            lower = np.nanpercentile(valid_global_y, 1)
            upper = np.nanpercentile(valid_global_y, 99)

            padding = 0.08 * (upper - lower)

            if not np.isfinite(padding) or padding == 0:
                padding = 1.0

            ylims = (
                lower - padding,
                upper + padding,
            )
        else:
            ylims = (-1, 1)

        fig, axes = plt.subplots(
            nrows=len(isi_values),
            ncols=1,
            figsize=(11, 2.9 * len(isi_values)),
            sharex=True,
            sharey=True,
        )

        if len(isi_values) == 1:
            axes = [axes]

        for ax, isi in zip(axes, isi_values):

            isi_data = participant_data[
                participant_data["isi_bin"] == isi
            ].copy()

            title_text = f"ISI {isi:.2f} s"

            for condition in ["NOSTIM", "STIM"]:

                condition_data = isi_data[
                    (isi_data["condition"] == condition)
                    & isi_data["reaction_time_valid"].astype(bool)
                    & np.isfinite(isi_data[y_col])
                    & np.isfinite(isi_data["trial_num"])
                ].sort_values("trial_num")

                if condition_data.empty:
                    continue

                # Always use the full session identifier from the config:
                # e.g. Lio_STIM or Lio_NOSTIM
                session_id = f"{participant}_{condition}"

                color = get_subject_color(
                    session_id,
                    subject_colors,
                )

                marker = "o" if condition == "STIM" else "s"

                ax.scatter(
                    condition_data["trial_num"],
                    condition_data[y_col],
                    color=color,
                    marker=marker,
                    label=condition,
                    edgecolor="black",
                    linewidth=0.25,
                    s=27,
                    alpha=0.60,
                    zorder=2,
                )

                x = condition_data[
                    "trial_num"
                ].to_numpy(dtype=float)

                y = condition_data[
                    y_col
                ].to_numpy(dtype=float)

                if len(x) >= 3 and np.nanstd(x) > 0:

                    regression = linregress(x, y)

                    x_fit = np.linspace(
                        xlim[0],
                        xlim[1],
                        300,
                    )

                    y_fit = (
                        regression.intercept
                        + regression.slope * x_fit
                    )

                    ax.plot(
                        x_fit,
                        y_fit,
                        color=color,
                        linewidth=2.8,
                        linestyle="-",
                        label=(
                            f"{condition} regression "
                            #f"(slope={regression.slope:.3f} ms/trial)"
                        ),
                        zorder=3,
                    )

            # Retrieve interaction test for this participant/ISI
            stats_row = trend_stats[
                (trend_stats["base_participant"] == participant)
                & np.isclose(
                    trend_stats["isi_bin"].astype(float),
                    float(isi),
                )
            ]

            if not stats_row.empty:

                stats_row = stats_row.iloc[0]

                stim_slope = stats_row[
                    "stim_slope_ms_per_trial"
                ]

                nostim_slope = stats_row[
                    "nostim_slope_ms_per_trial"
                ]

                slope_difference = stats_row[
                    "interaction_slope_difference_ms_per_trial"
                ]

                interaction_pvalue = stats_row[
                    "interaction_pvalue"
                ]

                title_text += (
                    f" | slope STIM={stim_slope:.3f}"
                    f", NOSTIM={nostim_slope:.3f}"
                    f" | Δslope={slope_difference:.3f}"
                    f" | p interaction={interaction_pvalue:.3g}"
                )

            ax.axhline(
                0,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.8,
                zorder=1,
            )

            if block_size is not None:
                for boundary in range(
                    block_size,
                    int(xlim[1]),
                    block_size,
                ):
                    ax.axvline(
                        boundary + 0.5,
                        color="gray",
                        linestyle=":",
                        linewidth=1,
                        alpha=0.7,
                        zorder=1,
                    )

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylims)
            ax.set_title(
                title_text,
                fontweight="bold",
                fontsize=10,
            )

            pretty_axes(ax)

        handles, labels = axes[0].get_legend_handles_labels()

        # Remove duplicate legend labels
        unique_legend = dict(zip(labels, handles))

        axes[0].legend(
            unique_legend.values(),
            unique_legend.keys(),
            frameon=False,
            ncol=2,
        )

        axes[-1].set_xlabel("Global trial number")
        fig.supylabel(y_label)

        fig.suptitle(
            f"{participant} — STIM versus NOSTIM RT trends",
            fontweight="bold",
            y=1.01,
        )

        save_pretty_fig(
            fig,
            f"{fig_prefix}_{participant}.png",
            plots_dir,
        )

def plot_stim_minus_nostim_rt_difference(
    paired_data,
    plots_dir,
    difference_col="rt_difference_centered_ms",
    y_label="STIM − NOSTIM centered RT (ms)",
    block_size=80,
    xlim=(0.5, 400.5),
    fig_prefix="stim_minus_nostim_rt_difference",
):
    """
    Plot the paired STIM minus NOSTIM RT difference.

    Negative values:
        RT is lower during STIM.

    Positive values:
        RT is higher during STIM.
    """

    for participant in paired_data["base_participant"].dropna().unique():

        participant_data = paired_data[
            paired_data["base_participant"] == participant
        ]

        isi_values = sorted(
            participant_data["isi_bin"].dropna().unique()
        )

        fig, axes = plt.subplots(
            len(isi_values),
            1,
            figsize=(11, 2.8 * len(isi_values)),
            sharex=True,
            sharey=True,
        )

        if len(isi_values) == 1:
            axes = [axes]

        for ax, isi in zip(axes, isi_values):

            isi_data = participant_data[
                (participant_data["isi_bin"] == isi)
                & participant_data["pair_valid"].astype(bool)
                & np.isfinite(participant_data[difference_col])
            ].sort_values("trial_num")

            if isi_data.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No valid paired data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            ax.scatter(
                isi_data["trial_num"],
                isi_data[difference_col],
                s=27,
                alpha=0.65,
                edgecolor="black",
                linewidth=0.2,
            )

            rolling_mean = isi_data[difference_col].rolling(
                window=10,
                min_periods=5,
                center=True,
            ).mean()

            ax.plot(
                isi_data["trial_num"],
                rolling_mean,
                linewidth=2.4,
            )

            if len(isi_data) >= 3:
                regression = linregress(
                    isi_data["trial_num"],
                    isi_data[difference_col],
                )

                x_fit = np.linspace(
                    isi_data["trial_num"].min(),
                    isi_data["trial_num"].max(),
                    100,
                )

                y_fit = (
                    regression.intercept
                    + regression.slope * x_fit
                )

                ax.plot(
                    x_fit,
                    y_fit,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                )

                title = (
                    f"ISI {isi:.2f} s"
                    f" | slope={regression.slope:.3f}"
                    f" | p={regression.pvalue:.3g}"
                )
            else:
                title = f"ISI {isi:.2f} s"

            ax.axhline(
                0,
                color="black",
                linestyle="-",
                linewidth=1,
            )

            for boundary in range(block_size, 400, block_size):
                ax.axvline(
                    boundary + 0.5,
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                    alpha=0.7,
                )

            ax.set_title(title, fontweight="bold")
            ax.set_xlim(*xlim)
            pretty_axes(ax)

        axes[-1].set_xlabel("Global trial number")
        fig.supylabel(y_label)

        fig.suptitle(
            f"{participant} — paired STIM − NOSTIM RT difference",
            fontweight="bold",
            y=1.01,
        )

        save_pretty_fig(
            fig,
            f"{fig_prefix}_{participant}.png",
            plots_dir,
        )

def plot_stim_nostim_rt_variability(
    long_data,
    variability_trend_stats,
    plots_dir,
    subject_colors,
    variability_col="reaction_time_rolling_sd_ms",
    block_size=80,
    xlim=(0.5, 400.5),
    fig_prefix="stim_vs_nostim_rt_variability",
):
    """
    Compare rolling RT variability between STIM and NOSTIM.

    One figure per participant:
        - one subplot per ISI
        - global trial number on x-axis
        - rolling RT SD on y-axis
        - one linear regression per condition
        - interaction p-value testing the difference between slopes

    Colors are taken from config["plot"]["subject_colors"].
    """

    long_data = long_data.copy()

    required_cols = {
        "base_participant",
        "condition",
        "isi_bin",
        "trial_num",
        variability_col,
    }

    missing = required_cols.difference(long_data.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    for participant in long_data[
        "base_participant"
    ].dropna().unique():

        participant_data = long_data[
            long_data["base_participant"] == participant
        ].copy()

        # Only keep participants with both conditions
        available_conditions = set(
            participant_data["condition"].dropna().unique()
        )

        if not {"STIM", "NOSTIM"}.issubset(available_conditions):
            continue

        isi_values = sorted(
            participant_data["isi_bin"]
            .dropna()
            .unique()
        )

        if not isi_values:
            continue

        valid_y = participant_data.loc[
            np.isfinite(participant_data[variability_col]),
            variability_col,
        ]

        if len(valid_y) > 0:
            lower = max(
                0,
                np.nanpercentile(valid_y, 1),
            )
            upper = np.nanpercentile(valid_y, 99)

            padding = 0.08 * (upper - lower)

            if not np.isfinite(padding) or padding == 0:
                padding = 1.0

            ylims = (
                max(0, lower - padding),
                upper + padding,
            )
        else:
            ylims = (0, 1)

        fig, axes = plt.subplots(
            nrows=len(isi_values),
            ncols=1,
            figsize=(11, 2.9 * len(isi_values)),
            sharex=True,
            sharey=True,
        )

        if len(isi_values) == 1:
            axes = [axes]

        for ax, isi in zip(axes, isi_values):

            isi_data = participant_data[
                participant_data["isi_bin"] == isi
            ].copy()

            title_text = f"ISI {isi:.2f} s"

            for condition in ["NOSTIM", "STIM"]:

                condition_data = isi_data[
                    (isi_data["condition"] == condition)
                    & np.isfinite(isi_data[variability_col])
                    & np.isfinite(isi_data["trial_num"])
                ].sort_values("trial_num")

                if condition_data.empty:
                    continue

                session_id = f"{participant}_{condition}"

                color = get_subject_color(
                    session_id,
                    subject_colors,
                )

                marker = "o" if condition == "STIM" else "s"

                ax.scatter(
                    condition_data["trial_num"],
                    condition_data[variability_col],
                    color=color,
                    marker=marker,
                    label=condition,
                    edgecolor="black",
                    linewidth=0.25,
                    s=24,
                    alpha=0.55,
                    zorder=2,
                )

                x = condition_data[
                    "trial_num"
                ].to_numpy(dtype=float)

                y = condition_data[
                    variability_col
                ].to_numpy(dtype=float)

                if len(x) >= 3 and np.nanstd(x) > 0:

                    regression = linregress(x, y)

                    x_fit = np.linspace(
                        xlim[0],
                        xlim[1],
                        300,
                    )

                    y_fit = (
                        regression.intercept
                        + regression.slope * x_fit
                    )

                    ax.plot(
                        x_fit,
                        y_fit,
                        color=color,
                        linewidth=2.8,
                        linestyle="-",
                        label=(
                            f"{condition} regression "
                            # f"(slope={regression.slope:.3f})"
                        ),
                        zorder=3,
                    )

            stats_row = variability_trend_stats[
                (
                    variability_trend_stats["base_participant"]
                    == participant
                )
                & np.isclose(
                    variability_trend_stats["isi_bin"].astype(float),
                    float(isi),
                )
            ]

            if not stats_row.empty:

                stats_row = stats_row.iloc[0]

                stim_slope = stats_row[
                    "stim_slope_sd_ms_per_trial"
                ]

                nostim_slope = stats_row[
                    "nostim_slope_sd_ms_per_trial"
                ]

                slope_difference = stats_row[
                    "interaction_slope_difference_sd_ms_per_trial"
                ]

                interaction_pvalue = stats_row[
                    "interaction_pvalue"
                ]

                title_text += (
                    f" | slope STIM={stim_slope:.3f}"
                    f", NOSTIM={nostim_slope:.3f}"
                    f" | Δslope={slope_difference:.3f}"
                    f" | p interaction={interaction_pvalue:.3g}"
                )

            for boundary in range(
                block_size,
                int(xlim[1]),
                block_size,
            ):
                ax.axvline(
                    boundary + 0.5,
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                    alpha=0.7,
                    zorder=1,
                )

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylims)
            ax.set_title(
                title_text,
                fontweight="bold",
                fontsize=10,
            )

            pretty_axes(ax)

        handles, labels = axes[0].get_legend_handles_labels()
        unique_legend = dict(zip(labels, handles))

        axes[0].legend(
            unique_legend.values(),
            unique_legend.keys(),
            frameon=False,
            ncol=2,
        )

        axes[-1].set_xlabel("Global trial number")
        fig.supylabel("Rolling RT variability, SD (ms)")

        fig.suptitle(
            f"{participant} — STIM versus NOSTIM RT variability trends",
            fontweight="bold",
            y=1.01,
        )

        save_pretty_fig(
            fig,
            f"{fig_prefix}_{participant}.png",
            plots_dir,
        )