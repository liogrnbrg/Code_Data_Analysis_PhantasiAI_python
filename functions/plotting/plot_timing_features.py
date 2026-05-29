import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from preprocessing.normalization import normalize_signal, get_normalization_label
from utils.colors import get_subject_color
from utils.style import pretty_axes, save_pretty_fig, get_robust_ylims
from utils.paths import make_safe_filename
from utils.config import get_config

config = get_config()
#plt.rcParams.update({'font.family': config["plot"]["font"]["family"]})

def _get_participants(timing_data):
    return timing_data["participant_id"].dropna().unique()


def _get_isi_values(timing_data):
    isi_col = "isi_bin" if "isi_bin" in timing_data.columns else "isi"
    return np.array(sorted(timing_data[isi_col].dropna().unique()), dtype=float), isi_col


def plot_peak_amplitude_regression_by_participant(
    timing_data,
    plots_dir,
    subject_colors,
    normalization="zscore",
):
    participants = _get_participants(timing_data)

    title_suffix, y_label = get_normalization_label(normalization)

    fig, axes = plt.subplots(
        nrows=len(participants),
        ncols=1,
        figsize=(10, max(4, 2.8 * len(participants))),
        sharex=True,
    )

    if len(participants) == 1:
        axes = [axes]

    all_y = []

    for participant_id in participants:
        dfp = timing_data[timing_data["participant_id"] == participant_id].sort_values("trial_num")
        y = normalize_signal(dfp["peak_amp"].to_numpy(), normalization)
        all_y.extend(y[np.isfinite(y)])

    ylims = get_robust_ylims(all_y)

    for ax, participant_id in zip(axes, participants):
        dfp = timing_data[timing_data["participant_id"] == participant_id].sort_values("trial_num")

        x = np.arange(1, len(dfp) + 1)
        y = normalize_signal(dfp["peak_amp"].to_numpy(), normalization)

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

            ax.plot(
                x,
                y_fit,
                color=color,
                linewidth=2.5,
            )

            ax.set_title(
                f"{participant_id} | slope = {res.slope:.4f}, p = {res.pvalue:.3g}"
            )
        else:
            ax.set_title(str(participant_id))

        ax.set_ylim(*ylims)
        pretty_axes(ax)

    fig.supxlabel("Trial number")
    fig.supylabel(y_label)
    fig.suptitle(f"Peak amplitude over trials — {title_suffix}", y=1.02)

    save_pretty_fig(
        fig,
        f"peak_amplitude_regression_{make_safe_filename(normalization)}.png",
        plots_dir,
    )


def plot_peak_amplitude_boxplot_by_isi(
    timing_data,
    plots_dir,
    subject_colors,
    normalization="zscore",
):
    participants = _get_participants(timing_data)
    isi_values, isi_col = _get_isi_values(timing_data)

    title_suffix, y_label = get_normalization_label(normalization)

    fig, axes = plt.subplots(
        nrows=len(participants),
        ncols=1,
        figsize=(10, max(4, 2.8 * len(participants))),
        sharex=True,
        sharey=True,
    )

    if len(participants) == 1:
        axes = [axes]

    all_y = []

    normalized_by_participant = {}

    for participant_id in participants:
        dfp = timing_data[timing_data["participant_id"] == participant_id].sort_values("trial_num").copy()
        y = normalize_signal(dfp["peak_amp"].to_numpy(), normalization)
        dfp["peak_amp_norm"] = y
        normalized_by_participant[participant_id] = dfp
        all_y.extend(y[np.isfinite(y)])

    ylims = get_robust_ylims(all_y)

    for ax, participant_id in zip(axes, participants):
        dfp = normalized_by_participant[participant_id]
        color = get_subject_color(participant_id, subject_colors)

        data_by_isi = []
        positions = np.arange(1, len(isi_values) + 1)

        for isi in isi_values:
            y = dfp.loc[dfp[isi_col] == isi, "peak_amp_norm"].to_numpy()
            data_by_isi.append(y[np.isfinite(y)])

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
            y = dfp.loc[dfp[isi_col] == isi, "peak_amp_norm"].to_numpy()
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
    fig.supylabel(y_label)
    fig.suptitle(f"Peak amplitude by ISI — {title_suffix}", y=1.02)

    save_pretty_fig(
        fig,
        f"peak_amplitude_boxplot_by_isi_{make_safe_filename(normalization)}.png",
        plots_dir,
    )


def plot_event_peak_delay_regression_by_participant(
    timing_data,
    plots_dir,
    subject_colors,
):
    participants = _get_participants(timing_data)

    fig, axes = plt.subplots(
        nrows=len(participants),
        ncols=1,
        figsize=(10, max(4, 2.8 * len(participants))),
        sharex=True,
    )

    if len(participants) == 1:
        axes = [axes]

    peak_delay = timing_data["peak_time"] - timing_data["event"]
    ylims = get_robust_ylims(peak_delay)

    for ax, participant_id in zip(axes, participants):
        dfp = timing_data[timing_data["participant_id"] == participant_id].sort_values("trial_num")

        x = np.arange(1, len(dfp) + 1)
        y = (dfp["peak_time"] - dfp["event"]).to_numpy()

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

            ax.plot(
                x,
                y_fit,
                color=color,
                linewidth=2.5,
            )

            ax.set_title(
                f"{participant_id} | slope = {res.slope:.4f}, p = {res.pvalue:.3g}"
            )
        else:
            ax.set_title(str(participant_id))

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(*ylims)
        pretty_axes(ax)

    fig.supxlabel("Trial number")
    fig.supylabel("Peak time - event time (s)")
    fig.suptitle("Delay between event and detected EMG peak over trials", y=1.02)

    save_pretty_fig(
        fig,
        "event_peak_delay_regression_by_participant.png",
        plots_dir,
    )


def plot_event_peak_delay_boxplot_by_isi(
    timing_data,
    plots_dir,
    subject_colors,
):
    participants = _get_participants(timing_data)
    isi_values, isi_col = _get_isi_values(timing_data)

    fig, axes = plt.subplots(
        nrows=len(participants),
        ncols=1,
        figsize=(10, max(4, 2.8 * len(participants))),
        sharex=True,
        sharey=True,
    )

    if len(participants) == 1:
        axes = [axes]

    all_delay = (timing_data["peak_time"] - timing_data["event"]).to_numpy()
    ylims = get_robust_ylims(all_delay)

    for ax, participant_id in zip(axes, participants):
        dfp = timing_data[timing_data["participant_id"] == participant_id].sort_values("trial_num").copy()
        dfp["peak_delay"] = dfp["peak_time"] - dfp["event"]

        color = get_subject_color(participant_id, subject_colors)

        data_by_isi = []
        positions = np.arange(1, len(isi_values) + 1)

        for isi in isi_values:
            y = dfp.loc[dfp[isi_col] == isi, "peak_delay"].to_numpy()
            data_by_isi.append(y[np.isfinite(y)])

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
            y = dfp.loc[dfp[isi_col] == isi, "peak_delay"].to_numpy()
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

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(*ylims)
        ax.set_title(str(participant_id))
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{isi:.2f}" for isi in isi_values])

        pretty_axes(ax)

    fig.supxlabel("ISI (s)")
    fig.supylabel("Peak time - event time (s)")
    fig.suptitle("Event-to-peak delay by ISI", y=1.02)

    save_pretty_fig(
        fig,
        "event_peak_delay_boxplot_by_isi.png",
        plots_dir,
    )

def plot_delta_peak_amplitude_vs_delta_isi_by_participant(
    timing_data,
    plots_dir,
    subject_colors,
):
    """
    Plot relative change in peak amplitude as a function of change in ISI.

    x = current ISI - previous ISI
    y = 100 * (current peak_amp - previous peak_amp) / previous peak_amp
    """

    participants = _get_participants(timing_data)

    # ---------- First pass: compute all values for shared y-limits ----------
    all_delta_amp = []

    delta_tables = {}

    for participant_id in participants:
        dfp = timing_data[timing_data["participant_id"] == participant_id].sort_values("trial_num").copy()

        if len(dfp) < 2:
            continue

        isi_col = "isi_bin" if "isi_bin" in dfp.columns else "isi"

        isi = dfp[isi_col].to_numpy(dtype=float)
        peak_amp = dfp["peak_amp"].to_numpy(dtype=float)

        delta_isi = isi[1:] - isi[:-1]
        delta_amp_pct = 100 * (peak_amp[1:] - peak_amp[:-1]) / peak_amp[:-1]

        valid = np.isfinite(delta_isi) & np.isfinite(delta_amp_pct)

        delta_isi = delta_isi[valid]
        delta_amp_pct = delta_amp_pct[valid]

        delta_tables[participant_id] = {
            "delta_isi": delta_isi,
            "delta_amp_pct": delta_amp_pct,
        }

        all_delta_amp.extend(delta_amp_pct)

    ylims = get_robust_ylims(all_delta_amp)

    # ---------- Figure ----------
    fig, axes = plt.subplots(
        nrows=len(participants),
        ncols=1,
        figsize=(10, max(4, 2.8 * len(participants))),
        sharex=False,
        sharey=True,
    )

    if len(participants) == 1:
        axes = [axes]

    for ax, participant_id in zip(axes, participants):

        color = get_subject_color(participant_id, subject_colors)

        if participant_id not in delta_tables:
            ax.text(
                0.5,
                0.5,
                "Not enough trials",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(str(participant_id))
            pretty_axes(ax)
            continue

        delta_isi = delta_tables[participant_id]["delta_isi"]
        delta_amp_pct = delta_tables[participant_id]["delta_amp_pct"]

        unique_delta_isi = np.array(sorted(np.unique(delta_isi)), dtype=float)
        positions = np.arange(1, len(unique_delta_isi) + 1)

        data_by_delta = []

        for d_isi in unique_delta_isi:
            y = delta_amp_pct[delta_isi == d_isi]
            data_by_delta.append(y[np.isfinite(y)])

        ax.boxplot(
            data_by_delta,
            positions=positions,
            widths=0.55,
            showfliers=False,
            patch_artist=False,
            boxprops=dict(color="black", linewidth=1.4),
            medianprops=dict(color="black", linewidth=1.7),
            whiskerprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1.2),
        )

        for i, d_isi in enumerate(unique_delta_isi, start=1):
            y = delta_amp_pct[delta_isi == d_isi]
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

        ax.axhline(0, color="black", linestyle="--", linewidth=1)

        ax.set_ylim(*ylims)
        ax.set_title(str(participant_id))

        ax.set_xticks(positions)
        ax.set_xticklabels([f"{d:.2f}" for d in unique_delta_isi])

        pretty_axes(ax)

    fig.supxlabel("Current ISI - previous ISI (s)")
    fig.supylabel("Relative change in peak amplitude (%)")
    fig.suptitle("Peak amplitude change as a function of previous ISI change", y=1.02)

    save_pretty_fig(
        fig,
        "delta_peak_amplitude_vs_delta_isi_by_participant.png",
        plots_dir,
    )

def plot_peak_amplitude_vs_recent_isi_context_by_participant(
    timing_data,
    plots_dir,
    subject_colors,
    n_previous_trials=3,
    normalization="zscore",
):
    """
    Scatter + regression.

    x = current ISI - mean(ISI over the previous n trials)
    y = current peak amplitude

    The first n trials are excluded because they do not have enough previous trials.
    """

    participants = _get_participants(timing_data)

    title_suffix, y_label = get_normalization_label(normalization)

    # ---------- First pass for shared y-limits ----------
    all_y = []
    all_x = []

    context_tables = {}

    for participant_id in participants:
        dfp = timing_data[timing_data["participant_id"] == participant_id].sort_values("trial_num").copy()

        isi_col = "isi_bin" if "isi_bin" in dfp.columns else "isi"

        isi = dfp[isi_col].to_numpy(dtype=float)
        peak_amp = dfp["peak_amp"].to_numpy(dtype=float)

        peak_amp_norm = normalize_signal(peak_amp, normalization)

        x_context = []
        y_current = []

        for i in range(n_previous_trials, len(dfp)):
            previous_isi = isi[i - n_previous_trials:i]

            if not np.all(np.isfinite(previous_isi)):
                continue

            previous_mean_isi = np.mean(previous_isi)

            x_context.append(isi[i] - previous_mean_isi)
            y_current.append(peak_amp_norm[i])

        x_context = np.asarray(x_context, dtype=float)
        y_current = np.asarray(y_current, dtype=float)

        valid = np.isfinite(x_context) & np.isfinite(y_current)

        x_context = x_context[valid]
        y_current = y_current[valid]

        context_tables[participant_id] = {
            "x_context": x_context,
            "y_current": y_current,
        }

        all_x.extend(x_context)
        all_y.extend(y_current)

    xlims = get_robust_ylims(all_x)
    ylims = get_robust_ylims(all_y)

    # ---------- Figure ----------
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
        color = get_subject_color(participant_id, subject_colors)

        x = context_tables[participant_id]["x_context"]
        y = context_tables[participant_id]["y_current"]

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

        ax.plot(
            x_fit,
            y_fit,
            color=color,
            linewidth=2.5,
        )

        ax.axvline(0, color="black", linestyle="--", linewidth=1)

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)

        ax.set_title(
            f"{participant_id} | slope = {res.slope:.4f}, "
            f"r = {res.rvalue:.3f}, p = {res.pvalue:.3g}"
        )

        pretty_axes(ax)

    fig.supxlabel(
        f"Current ISI - mean ISI over previous {n_previous_trials} trials (s)"
    )
    fig.supylabel(y_label)
    fig.suptitle(
        f"Peak amplitude vs recent ISI context — {title_suffix}",
        y=1.02,
    )

    save_pretty_fig(
        fig,
        f"peak_amplitude_vs_recent_{n_previous_trials}_trial_isi_context_{make_safe_filename(normalization)}.png",
        plots_dir,
    )

def plot_current_peak_amplitude_vs_delta_isi_by_participant(
    timing_data,
    plots_dir,
    subject_colors,
    normalization="zscore",
):
    """
    Scatter + regression.

    x = current ISI - previous ISI
    y = current peak amplitude

    The first trial is excluded because it has no previous trial.
    """

    participants = _get_participants(timing_data)

    title_suffix, y_label = get_normalization_label(normalization)

    # ---------- First pass for shared limits ----------
    all_x = []
    all_y = []

    participant_tables = {}

    for participant_id in participants:

        dfp = timing_data[
            timing_data["participant_id"] == participant_id
        ].sort_values("trial_num").copy()

        if len(dfp) < 2:
            continue

        isi_col = "isi_bin" if "isi_bin" in dfp.columns else "isi"

        isi = dfp[isi_col].to_numpy(dtype=float)
        peak_amp = dfp["peak_amp"].to_numpy(dtype=float)

        peak_amp_norm = normalize_signal(peak_amp, normalization)

        delta_isi = isi[1:] - isi[:-1]
        current_peak_amp = peak_amp_norm[1:]

        valid = np.isfinite(delta_isi) & np.isfinite(current_peak_amp)

        delta_isi = delta_isi[valid]
        current_peak_amp = current_peak_amp[valid]

        participant_tables[participant_id] = {
            "delta_isi": delta_isi,
            "current_peak_amp": current_peak_amp,
        }

        all_x.extend(delta_isi)
        all_y.extend(current_peak_amp)

    xlims = get_robust_ylims(all_x)
    ylims = get_robust_ylims(all_y)

    # ---------- Figure ----------
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

        color = get_subject_color(participant_id, subject_colors)

        if participant_id not in participant_tables:
            ax.text(
                0.5,
                0.5,
                "Not enough trials",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(str(participant_id))
            pretty_axes(ax)
            continue

        x = participant_tables[participant_id]["delta_isi"]
        y = participant_tables[participant_id]["current_peak_amp"]

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

        ax.plot(
            x_fit,
            y_fit,
            color=color,
            linewidth=2.5,
        )

        ax.axvline(0, color="black", linestyle="--", linewidth=1)

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)

        ax.set_title(
            f"{participant_id} | slope = {res.slope:.4f}, "
            f"r = {res.rvalue:.3f}, p = {res.pvalue:.3g}"
        )

        pretty_axes(ax)

    fig.supxlabel("Current ISI - previous ISI (s)")
    fig.supylabel(y_label)
    fig.suptitle(
        f"Current peak amplitude vs previous ISI change — {title_suffix}",
        y=1.02,
    )

    save_pretty_fig(
        fig,
        f"current_peak_amplitude_vs_delta_isi_{make_safe_filename(normalization)}.png",
        plots_dir,
    )