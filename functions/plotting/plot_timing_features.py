import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels.api as sm

from preprocessing.normalization import normalize_signal, get_normalization_label
from utils.colors import get_subject_color
from utils.style import pretty_axes, save_pretty_fig, get_robust_ylims
from utils.paths import make_safe_filename
from utils.config import get_config

config = get_config()
#plt.rcParams.update({'font.family': config["plot"]["font"]["family"]})

def _get_participants(timing_data):
    return timing_data["participant_id"].dropna().unique()
def _pvalue_to_stars(pvalue):
    """
    Convert a p-value to significance stars.
    """
    if not np.isfinite(pvalue):
        return "NA"
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def _format_combined_regression_legend_label(
    participant_id,
    slope,
    pvalue,
):
    """
    Create a compact legend label for combined regression plots.
    """
    stars = _pvalue_to_stars(pvalue)

    return (
        f"{participant_id}   "
        f"\u03b2={slope:+.4f}   "
        f"p={pvalue:.3g} {stars}"
    )


def _format_band_description(
    band,
    sd_multiplier=1.0,
    ci_level=0.95,
):
    """
    Human-readable band description for titles / legend titles.
    """
    if band is None:
        return "No band"

    if band == "sd":
        if sd_multiplier == 1:
            return "\u00b11 SD band"
        return f"\u00b1{sd_multiplier:g} SD band"

    if band == "ci":
        return f"{int(round(ci_level * 100))}% CI band"

    return str(band)


def _y_limits_from_regressions(
    regressions,
):
    """
    Compute y-limits from regression lines and optional bands.
    """
    ymins = []
    ymaxs = []

    for item in regressions:
        reg = item["regression"]

        y_fit = np.asarray(reg["y_fit"], dtype=float)
        ymins.append(np.nanmin(y_fit))
        ymaxs.append(np.nanmax(y_fit))

        band_lower = reg.get("band_lower", None)
        band_upper = reg.get("band_upper", None)

        if band_lower is not None and band_upper is not None:
            ymins.append(np.nanmin(band_lower))
            ymaxs.append(np.nanmax(band_upper))

    ymin = np.nanmin(ymins)
    ymax = np.nanmax(ymaxs)

    yrange = ymax - ymin
    if yrange == 0:
        yrange = 1.0

    pad = 0.08 * yrange

    return ymin - pad, ymax + pad


def _plot_combined_regressions(
    regressions,
    plots_dir,
    subject_colors,
    filename,
    title,
    y_label,
    alpha=0.05,
    band="sd",
    sd_multiplier=1.0,
    ci_level=0.95,
    show_zero_line=False,
):
    """
    Generic combined regression plot.

    Parameters
    ----------
    regressions : list of dict
        Each item must contain:
            "participant_id"
            "regression"
    """

    if not regressions:
        print(f"No regressions available for {filename}.")
        return

    # Sort: significant first, then by participant name
    for item in regressions:
        pvalue = item["regression"]["slope_pvalue"]
        item["is_significant"] = (
            np.isfinite(pvalue)
            and pvalue < alpha
        )

    regressions = sorted(
        regressions,
        key=lambda d: (
            not d["is_significant"],
            d["participant_id"],
        ),
    )

    fig, ax = plt.subplots(
        figsize=(12.5, 7),
    )

    # Draw bands first, then lines
    for item in regressions:
        participant_id = item["participant_id"]
        regression = item["regression"]
        is_significant = item["is_significant"]

        color = get_subject_color(
            participant_id,
            subject_colors,
        )

        band_lower = regression.get("band_lower", None)
        band_upper = regression.get("band_upper", None)

        if band_lower is not None and band_upper is not None:
            ax.fill_between(
                regression["x_fit"],
                band_lower,
                band_upper,
                color=color,
                alpha=0.12 if is_significant else 0.06,
                linewidth=0,
                zorder=1,
            )

    for item in regressions:
        participant_id = item["participant_id"]
        regression = item["regression"]
        is_significant = item["is_significant"]

        color = get_subject_color(
            participant_id,
            subject_colors,
        )

        pvalue = regression["slope_pvalue"]

        ax.plot(
            regression["x_fit"],
            regression["y_fit"],
            color=color,
            linewidth=3.2 if is_significant else 2.0,
            linestyle="-" if is_significant else "--",
            alpha=0.95 if is_significant else 0.65,
            zorder=3 if is_significant else 2,
            label=_format_combined_regression_legend_label(
                participant_id=participant_id,
                slope=regression["slope"],
                pvalue=pvalue,
            ),
        )

    # Limits
    ymin, ymax = _y_limits_from_regressions(regressions)
    ax.set_ylim(ymin, ymax)

    # Show y=0 only if it is relevant
    if show_zero_line and ymin <= 0 <= ymax:
        ax.axhline(
            0,
            color="black",
            linestyle=":",
            linewidth=1.2,
            alpha=0.6,
            zorder=0,
        )

    ax.set_xlabel("Trial number")
    ax.set_ylabel(y_label)

    band_description = _format_band_description(
        band=band,
        sd_multiplier=sd_multiplier,
        ci_level=ci_level,
    )

    ax.set_title(
        f"{title}\n"
        f"Solid = p < {alpha:.2f} | Dashed = ns | "
        f"Shaded = {band_description}"
    )

    pretty_axes(ax)

    # Put legend outside
    legend_title = (
        "Participant-level HAC regressions"
    )

    ax.legend(
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        right=0.72,
    )

    save_pretty_fig(
        fig,
        filename,
        plots_dir,
    )

def _get_isi_values(timing_data):
    isi_col = "isi_bin" if "isi_bin" in timing_data.columns else "isi"
    return np.array(sorted(timing_data[isi_col].dropna().unique()), dtype=float), isi_col

def fit_linear_regression(
    x,
    y,
    x_fit=None,
    band="sd",
    ci_level=0.95,
    sd_multiplier=1.0,
    hac_maxlags=10,
):
    """
    Fit a linear regression and calculate either:

    - an SD band around the fitted line, or
    - a confidence interval around the estimated mean regression line.

    Parameters
    ----------
    x, y
        Observed values.

    x_fit
        X values at which predictions are calculated. If None, 200 values
        spanning the observed x range are used.

    band : {"sd", "ci", None}
        Type of band to calculate.

        "sd"
            Fitted line ± residual standard deviation.
            This is the default.

        "ci"
            Confidence interval around the estimated mean regression line.

        None
            Do not calculate a band.

    ci_level
        Confidence level used when band="ci".

    sd_multiplier
        Number of residual standard deviations shown when band="sd".
        For example:
            1.0 = ±1 SD
            2.0 = ±2 SD

    hac_maxlags
        If provided, use HAC robust standard errors with this number of lags.
        Set to None to use ordinary OLS standard errors.

    Returns
    -------
    dict containing:
        x_fit
        y_fit
        band_lower
        band_upper
        band_type
        intercept
        slope
        slope_pvalue
        rvalue
        r_squared
        residual_sd
    """

    if band not in {"sd", "ci", None}:
        raise ValueError(
            f"band must be 'sd', 'ci', or None; received {band!r}."
        )

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3 or np.nanstd(x) == 0:
        return None

    if x_fit is None:
        x_fit = np.linspace(
            np.nanmin(x),
            np.nanmax(x),
            200,
        )
    else:
        x_fit = np.asarray(x_fit, dtype=float)

    X = sm.add_constant(
        x,
        has_constant="add",
    )

    model = sm.OLS(y, X)

    if hac_maxlags is None:
        result = model.fit()
    else:
        result = model.fit(
            cov_type="HAC",
            cov_kwds={"maxlags": hac_maxlags},
        )

    X_fit = sm.add_constant(
        x_fit,
        has_constant="add",
    )

    # Fitted regression line
    y_fit = np.asarray(
        result.predict(X_fit),
        dtype=float,
    )

    # Residual SD based on the observed values around the fitted line.
    residuals = y - np.asarray(result.predict(X), dtype=float)
    residual_sd = np.sqrt(
        np.sum(residuals**2) / result.df_resid
    )

    if band == "sd":
        band_lower = y_fit - sd_multiplier * residual_sd
        band_upper = y_fit + sd_multiplier * residual_sd

    elif band == "ci":
        prediction = result.get_prediction(X_fit)

        prediction_frame = prediction.summary_frame(
            alpha=1 - ci_level,
        )

        band_lower = prediction_frame[
            "mean_ci_lower"
        ].to_numpy()

        band_upper = prediction_frame[
            "mean_ci_upper"
        ].to_numpy()

    else:
        band_lower = None
        band_upper = None

    correlation = np.corrcoef(x, y)[0, 1]

    return {
        "x_fit": x_fit,
        "y_fit": y_fit,
        "band_lower": band_lower,
        "band_upper": band_upper,
        "band_type": band,
        "intercept": result.params[0],
        "slope": result.params[1],
        "slope_pvalue": result.pvalues[1],
        "rvalue": correlation,
        "r_squared": result.rsquared,
        "residual_sd": residual_sd,
    }

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
            regression = fit_linear_regression(
                x=x[valid],
                y=y[valid],
                x_fit=np.linspace(
                    np.nanmin(x[valid]),
                    np.nanmax(x[valid]),
                    200,
                ),
                band="sd",
                hac_maxlags=10,
            )

            if regression is not None:

                ax.fill_between(
                    regression["x_fit"],
                    regression["band_lower"],
                    regression["band_upper"],
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                    zorder=1,
                )

                ax.plot(
                    regression["x_fit"],
                    regression["y_fit"],
                    color=color,
                    linewidth=2.5,
                    zorder=3,
                )

                ax.set_title(
                    f"{participant_id} | "
                    f"slope={regression['slope']:.4f}, "
                    f"p={regression['slope_pvalue']:.3g}"
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
            regression = fit_linear_regression(
                x=x[valid],
                y=y[valid],
                x_fit=np.linspace(
                    np.nanmin(x[valid]),
                    np.nanmax(x[valid]),
                    200,
                ),
                band="sd",
                hac_maxlags=10,
            )

            if regression is not None:
                ax.fill_between(
                    regression["x_fit"],
                    regression["band_lower"],
                    regression["band_upper"],
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                    zorder=1,
                )

                ax.plot(
                    regression["x_fit"],
                    regression["y_fit"],
                    color=color,
                    linewidth=2.5,
                    zorder=3,
                )

                ax.set_title(
                    f"{participant_id} | "
                    f"slope = {regression['slope']:.4f}, "
                    f"p = {regression['slope_pvalue']:.3g}"
                )
            else:
                ax.set_title(str(participant_id))
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

def plot_delta_peak_amplitude_by_delta_isi_sign_by_participant(
    timing_data,
    plots_dir,
    subject_colors,
    zero_tolerance=1e-9,
):
    """
    Boxplots of relative change in peak amplitude grouped by the sign of delta ISI.

    Groups:
        Negative  -> current ISI - previous ISI < 0
        No change -> current ISI - previous ISI == 0
        Positive  -> current ISI - previous ISI > 0

    y = 100 * (current peak_amp - previous peak_amp) / previous peak_amp
    """

    participants = _get_participants(timing_data)

    all_delta_amp = []
    delta_tables = {}

    for participant_id in participants:
        dfp = timing_data[
            timing_data["participant_id"] == participant_id
        ].sort_values("trial_num").copy()

        if len(dfp) < 2:
            continue

        isi_col = "isi_bin" if "isi_bin" in dfp.columns else "isi"

        isi = dfp[isi_col].to_numpy(dtype=float)
        peak_amp = dfp["peak_amp"].to_numpy(dtype=float)

        delta_isi = isi[1:] - isi[:-1]
        delta_amp_pct = 100 * (peak_amp[1:] - peak_amp[:-1]) / peak_amp[:-1]

        valid = (
            np.isfinite(delta_isi)
            & np.isfinite(delta_amp_pct)
            & np.isfinite(peak_amp[:-1])
            & (peak_amp[:-1] != 0)
        )

        delta_isi = delta_isi[valid]
        delta_amp_pct = delta_amp_pct[valid]

        negative = delta_amp_pct[delta_isi < -zero_tolerance]
        no_change = delta_amp_pct[np.abs(delta_isi) <= zero_tolerance]
        positive = delta_amp_pct[delta_isi > zero_tolerance]

        delta_tables[participant_id] = {
            "Negative": negative,
            "No change": no_change,
            "Positive": positive,
        }

        all_delta_amp.extend(delta_amp_pct)

    ylims = get_robust_ylims(all_delta_amp) if len(all_delta_amp) > 0 else (-1, 1)

    fig, axes = plt.subplots(
        nrows=len(participants),
        ncols=1,
        figsize=(10, max(4, 2.8 * len(participants))),
        sharex=True,
        sharey=True,
    )

    if len(participants) == 1:
        axes = [axes]

    labels = ["Negative", "No change", "Positive"]
    positions = [1, 2, 3]

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

        grouped = delta_tables[participant_id]

        for pos, label in zip(positions, labels):
            y = grouped[label]
            y = y[np.isfinite(y)]

            if len(y) > 0:
                ax.boxplot(
                    [y],
                    positions=[pos],
                    widths=0.55,
                    showfliers=False,
                    patch_artist=False,
                    boxprops=dict(color="black", linewidth=1.4),
                    medianprops=dict(color="black", linewidth=1.7),
                    whiskerprops=dict(color="black", linewidth=1.2),
                    capprops=dict(color="black", linewidth=1.2),
                )

                x = pos + (np.random.rand(len(y)) - 0.5) * 0.20

                ax.scatter(
                    x,
                    y,
                    s=24,
                    color=color,
                    edgecolor="black",
                    linewidth=0.25,
                    alpha=0.65,
                )
            else:
                ax.text(
                    pos,
                    np.mean(ylims),
                    "n=0",
                    ha="center",
                    va="center",
                    fontsize=9,
                    alpha=0.7,
                )

        ax.axhline(0, color="black", linestyle="--", linewidth=1)

        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(*ylims)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(str(participant_id))

        pretty_axes(ax)

    fig.supxlabel("Direction of ISI change")
    fig.supylabel("Relative change in peak amplitude (%)")
    fig.suptitle(
        "Peak amplitude change grouped by the direction of ISI change",
        y=1.02,
    )

    save_pretty_fig(
        fig,
        "delta_peak_amplitude_by_delta_isi_sign_by_participant.png",
        plots_dir,
    )

def plot_delta_peak_amplitude_vs_delta_isi_regression_by_participant(
    timing_data,
    plots_dir,
    subject_colors,
    band="sd",
):
    """
    Scatter + regression.

    x = current ISI - previous ISI
    y = 100 * (current peak_amp - previous peak_amp) / previous peak_amp
    """

    participants = _get_participants(timing_data)

    all_x = []
    all_y = []
    delta_tables = {}

    for participant_id in participants:
        dfp = timing_data[
            timing_data["participant_id"] == participant_id
        ].sort_values("trial_num").copy()

        if len(dfp) < 2:
            continue

        isi_col = "isi_bin" if "isi_bin" in dfp.columns else "isi"

        isi = dfp[isi_col].to_numpy(dtype=float)
        peak_amp = dfp["peak_amp"].to_numpy(dtype=float)

        delta_isi = isi[1:] - isi[:-1]
        delta_amp_pct = 100 * (peak_amp[1:] - peak_amp[:-1]) / peak_amp[:-1]

        valid = (
            np.isfinite(delta_isi)
            & np.isfinite(delta_amp_pct)
            & np.isfinite(peak_amp[:-1])
            & (peak_amp[:-1] != 0)
        )

        delta_isi = delta_isi[valid]
        delta_amp_pct = delta_amp_pct[valid]

        delta_tables[participant_id] = {
            "delta_isi": delta_isi,
            "delta_amp_pct": delta_amp_pct,
        }

        all_x.extend(delta_isi)
        all_y.extend(delta_amp_pct)

    xlims = get_robust_ylims(all_x)
    ylims = get_robust_ylims(all_y)

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

        x = delta_tables[participant_id]["delta_isi"]
        y = delta_tables[participant_id]["delta_amp_pct"]

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

        regression = fit_linear_regression(
            x=x,
            y=y,
            x_fit=np.linspace(
                np.nanmin(x),
                np.nanmax(x),
                200,
            ),
            band=band,
            hac_maxlags=10,
        )

        if regression is not None:
            if (
                regression["band_lower"] is not None
                and regression["band_upper"] is not None
            ):
                ax.fill_between(
                    regression["x_fit"],
                    regression[f"band_lower"],
                    regression[f"band_upper"],
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                    zorder=1,
                )

                ax.plot(
                    regression["x_fit"],
                    regression["y_fit"],
                    color=color,
                    linewidth=2.5,
                    zorder=3,
                )

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)

        ax.set_title(
            f"{participant_id} | slope = {regression['slope']:.4f}, "
            f"r = {regression['rvalue']:.3f}, "
            f"p = {regression['slope_pvalue']:.3g}"
        )

        pretty_axes(ax)

    fig.supxlabel("Current ISI - previous ISI (s)")
    fig.supylabel("Relative change in peak amplitude (%)")
    fig.suptitle(
        "Peak amplitude change vs previous ISI change",
        y=1.02,
    )

    save_pretty_fig(
        fig,
        "delta_peak_amplitude_vs_delta_isi_regression_by_participant.png",
        plots_dir,
    )

def plot_peak_amplitude_vs_recent_isi_context_by_participant(
    timing_data,
    plots_dir,
    subject_colors,
    n_previous_trials=3,
    normalization="zscore",
    band="sd",
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

        regression = fit_linear_regression(
            x=x,
            y=y,
            x_fit=np.linspace(
                np.nanmin(x),
                np.nanmax(x),
                200,
            ),
            band=band,
            hac_maxlags=10,
        )

        if regression is not None:
            if (
                regression["band_lower"] is not None
                and regression["band_upper"] is not None
            ):
                ax.fill_between(
                    regression["x_fit"],
                    regression[f"band_lower"],
                    regression[f"band_upper"],
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                    zorder=1,
                )

                ax.plot(
                    regression["x_fit"],
                    regression["y_fit"],
                    color=color,
                    linewidth=2.5,
                    zorder=3,
                )

        ax.axvline(0, color="black", linestyle="--", linewidth=1)

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)

        if regression is not None:
            ax.set_title(
                f"{participant_id} | "
                f"slope = {regression['slope']:.4f}, "
                f"r = {regression['rvalue']:.3f}, "
                f"p = {regression['slope_pvalue']:.3g}"
            )
        else:
            ax.set_title(str(participant_id))

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
    band="sd",
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

        regression = fit_linear_regression(
            x=x,
            y=y,
            x_fit=np.linspace(
                np.nanmin(x),
                np.nanmax(x),
                200,
            ),
            band=band,
            hac_maxlags=10,
        )

        if regression is not None:
            ax.fill_between(
                regression["x_fit"],
                regression[f"band_lower"],
                regression[f"band_upper"],
                color=color,
                alpha=0.20,
                linewidth=0,
                zorder=1,
            )

            ax.plot(
                regression["x_fit"],
                regression["y_fit"],
                color=color,
                linewidth=2.5,
                zorder=3,
            )

        ax.axvline(0, color="black", linestyle="--", linewidth=1)

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)

        if regression is not None:
            ax.set_title(
                f"{participant_id} | slope = {regression['slope']:.4f}, "
                f"r = {regression['rvalue']:.3f}, p = {regression['slope_pvalue']:.3g}"
            )
        else:
            ax.set_title(str(participant_id))

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
def plot_event_peak_delay_regressions_combined(
    timing_data,
    plots_dir,
    subject_colors,
    alpha=0.05,
    hac_maxlags=10,
    band="sd",
    sd_multiplier=1.0,
    ci_level=0.95,
):
    """
    Plot all participant-level event-to-peak-delay regressions
    on one axis, with optional shaded band.

    Parameters
    ----------
    band : {"sd", "ci", None}
        Band shown around each regression line.
        Default is "sd".

    sd_multiplier : float
        Number of SDs used when band="sd".

    ci_level : float
        Confidence level used when band="ci".
    """

    participants = _get_participants(timing_data)
    regressions = []

    for participant_id in participants:
        dfp = (
            timing_data[
                timing_data["participant_id"] == participant_id
            ]
            .sort_values("trial_num")
            .copy()
        )

        x = np.arange(
            1,
            len(dfp) + 1,
            dtype=float,
        )

        y = (
            dfp["peak_time"].to_numpy(dtype=float)
            - dfp["event"].to_numpy(dtype=float)
        )

        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() < 3:
            continue

        regression = fit_linear_regression(
            x=x[valid],
            y=y[valid],
            x_fit=np.linspace(
                np.nanmin(x[valid]),
                np.nanmax(x[valid]),
                200,
            ),
            band=band,
            sd_multiplier=sd_multiplier,
            ci_level=ci_level,
            hac_maxlags=hac_maxlags,
        )

        if regression is None:
            continue

        regressions.append(
            {
                "participant_id": participant_id,
                "regression": regression,
            }
        )

    if not regressions:
        print(
            "No valid participant regressions available for "
            "event-to-peak delay."
        )
        return

    _plot_combined_regressions(
        regressions=regressions,
        plots_dir=plots_dir,
        subject_colors=subject_colors,
        filename="event_peak_delay_regressions_combined.png",
        title="Participant regression lines for event-to-peak delay",
        y_label="Peak time - event time (s)",
        alpha=alpha,
        band=band,
        sd_multiplier=sd_multiplier,
        ci_level=ci_level,
        show_zero_line=False,
    )

def plot_peak_amplitude_regressions_combined(
    timing_data,
    plots_dir,
    subject_colors,
    normalization="zscore",
    alpha=0.05,
    hac_maxlags=10,
    band="sd",
    sd_multiplier=1.0,
    ci_level=0.95,
):
    """
    Plot all participant-level peak-amplitude regressions
    on one axis, with optional shaded band.

    Parameters
    ----------
    normalization : {"raw", "first", "zscore"}
        Peak-amplitude normalization mode.

    band : {"sd", "ci", None}
        Band shown around each regression line.
        Default is "sd".

    sd_multiplier : float
        Number of SDs used when band="sd".

    ci_level : float
        Confidence level used when band="ci".
    """

    participants = _get_participants(timing_data)

    title_suffix, y_label = get_normalization_label(
        normalization
    )

    regressions = []

    for participant_id in participants:
        dfp = (
            timing_data[
                timing_data["participant_id"] == participant_id
            ]
            .sort_values("trial_num")
            .copy()
        )

        x = np.arange(
            1,
            len(dfp) + 1,
            dtype=float,
        )

        y = normalize_signal(
            dfp["peak_amp"].to_numpy(dtype=float),
            normalization,
        )

        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() < 3:
            continue

        regression = fit_linear_regression(
            x=x[valid],
            y=y[valid],
            x_fit=np.linspace(
                np.nanmin(x[valid]),
                np.nanmax(x[valid]),
                200,
            ),
            band=band,
            sd_multiplier=sd_multiplier,
            ci_level=ci_level,
            hac_maxlags=hac_maxlags,
        )

        if regression is None:
            continue

        regressions.append(
            {
                "participant_id": participant_id,
                "regression": regression,
            }
        )

    if not regressions:
        print(
            "No valid participant regressions available for "
            f"peak amplitude, normalization={normalization!r}."
        )
        return

    _plot_combined_regressions(
        regressions=regressions,
        plots_dir=plots_dir,
        subject_colors=subject_colors,
        filename=(
            "peak_amplitude_regressions_combined_"
            f"{make_safe_filename(normalization)}.png"
        ),
        title=(
            "Participant regression lines for peak amplitude — "
            f"{title_suffix}"
        ),
        y_label=y_label,
        alpha=alpha,
        band=band,
        sd_multiplier=sd_multiplier,
        ci_level=ci_level,
        show_zero_line=False,
    )