import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from utils.colors import get_subject_color
from utils.style import pretty_axes, save_pretty_fig, get_robust_ylims
from utils.paths import make_safe_filename


def plot_behavior_metric_over_trials(
    behavior_data,
    metric,
    plots_dir,
    subject_colors,
):
    """
    Plot a behavioral metric over trials for each participant.
    """

    participants = behavior_data["participant_id"].dropna().unique()

    fig, axes = plt.subplots(
        nrows=len(participants),
        ncols=1,
        figsize=(10, max(4, 2.8 * len(participants))),
        sharex=True,
        sharey=True,
    )

    if len(participants) == 1:
        axes = [axes]

    ylims = get_robust_ylims(behavior_data[metric])

    for ax, participant_id in zip(axes, participants):

        dfp = behavior_data[
            behavior_data["participant_id"] == participant_id
        ].sort_values("trial_num")

        x = dfp["trial_num"].to_numpy(dtype=float)
        y = dfp[metric].to_numpy(dtype=float)

        color = get_subject_color(participant_id, subject_colors)

        ax.scatter(
            x,
            y,
            s=28,
            color=color,
            edgecolor="black",
            linewidth=0.3,
            alpha=0.70,
        )

        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() >= 3:
            res = linregress(x[valid], y[valid])
            y_fit = res.intercept + res.slope * x

            ax.plot(x, y_fit, color=color, linewidth=2.5)

            ax.set_title(
                f"{participant_id} | slope = {res.slope:.4f}, "
                f"r = {res.rvalue:.3f}, p = {res.pvalue:.3g}"
            )
        else:
            ax.set_title(str(participant_id))

        ax.set_ylim(*ylims)
        pretty_axes(ax)

    fig.supxlabel("Trial number")
    fig.supylabel(metric.replace("_", " "))
    fig.suptitle(f"{metric.replace('_', ' ')} over trials", y=1.02)

    save_pretty_fig(
        fig,
        f"{make_safe_filename(metric)}_over_trials.png",
        plots_dir,
    )


def plot_behavior_metric_by_isi(
    behavior_data,
    metric,
    plots_dir,
    subject_colors,
):
    """
    Boxplot of a behavioral metric by ISI for each participant.
    """

    participants = behavior_data["participant_id"].dropna().unique()
    isi_values = np.array(sorted(behavior_data["isi_bin"].dropna().unique()), dtype=float)

    fig, axes = plt.subplots(
        nrows=len(participants),
        ncols=1,
        figsize=(10, max(4, 2.8 * len(participants))),
        sharex=True,
        sharey=True,
    )

    if len(participants) == 1:
        axes = [axes]

    ylims = get_robust_ylims(behavior_data[metric])

    for ax, participant_id in zip(axes, participants):

        dfp = behavior_data[
            behavior_data["participant_id"] == participant_id
        ].sort_values("trial_num")

        color = get_subject_color(participant_id, subject_colors)

        data_by_isi = []
        positions = np.arange(1, len(isi_values) + 1)

        for isi in isi_values:
            y = dfp.loc[dfp["isi_bin"] == isi, metric].to_numpy(dtype=float)
            y = y[np.isfinite(y)]
            data_by_isi.append(y)

        ax.boxplot(
            data_by_isi,
            positions=positions,
            widths=0.55,
            showfliers=False,
            patch_artist=False,
            boxprops=dict(color="black", linewidth=1.4),
            medianprops=dict(color="black", linewidth=1.7),
            whiskerprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1.2),
        )

        for i, isi in enumerate(isi_values, start=1):
            y = dfp.loc[dfp["isi_bin"] == isi, metric].to_numpy(dtype=float)
            y = y[np.isfinite(y)]

            x = i + (np.random.rand(len(y)) - 0.5) * 0.20

            ax.scatter(
                x,
                y,
                s=24,
                color=color,
                edgecolor="black",
                linewidth=0.25,
                alpha=0.65,
            )

        ax.set_ylim(*ylims)
        ax.set_title(str(participant_id))
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{isi:.2f}" for isi in isi_values])

        pretty_axes(ax)

    fig.supxlabel("ISI (s)")
    fig.supylabel(metric.replace("_", " "))
    fig.suptitle(f"{metric.replace('_', ' ')} by ISI", y=1.02)

    save_pretty_fig(
        fig,
        f"{make_safe_filename(metric)}_by_isi.png",
        plots_dir,
    )


def plot_emg_vs_behavior_metric(
    merged_data,
    emg_metric,
    behavior_metric,
    plots_dir,
    subject_colors,
):
    """
    Scatter + regression between an EMG metric and a behavioral metric.
    """

    participants = merged_data["participant_id"].dropna().unique()

    xlims = get_robust_ylims(merged_data[behavior_metric])
    ylims = get_robust_ylims(merged_data[emg_metric])

    fig, axes = plt.subplots(
        nrows=len(participants),
        ncols=1,
        figsize=(10, max(4, 2.8 * len(participants))),
        sharex=True,
        sharey=True,
    )

    if len(participants) == 1:
        axes = [axes]

    for ax, participant_id in zip(axes, participants):

        dfp = merged_data[
            merged_data["participant_id"] == participant_id
        ].copy()

        x = dfp[behavior_metric].to_numpy(dtype=float)
        y = dfp[emg_metric].to_numpy(dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)

        x = x[valid]
        y = y[valid]

        color = get_subject_color(participant_id, subject_colors)

        if len(x) < 3:
            ax.text(
                0.5,
                0.5,
                "Not enough valid trials",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(str(participant_id))
            pretty_axes(ax)
            continue

        ax.scatter(
            x,
            y,
            s=30,
            color=color,
            edgecolor="black",
            linewidth=0.3,
            alpha=0.70,
        )

        res = linregress(x, y)

        x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        y_fit = res.intercept + res.slope * x_fit

        ax.plot(x_fit, y_fit, color=color, linewidth=2.5)

        ax.set_title(
            f"{participant_id} | r = {res.rvalue:.3f}, p = {res.pvalue:.3g}"
        )

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)

        pretty_axes(ax)

    fig.supxlabel(behavior_metric.replace("_", " "))
    fig.supylabel(emg_metric.replace("_", " "))
    fig.suptitle(
        f"{emg_metric.replace('_', ' ')} vs {behavior_metric.replace('_', ' ')}",
        y=1.02,
    )

    save_pretty_fig(
        fig,
        f"{make_safe_filename(emg_metric)}_vs_{make_safe_filename(behavior_metric)}.png",
        plots_dir,
    )