# plot_reaction_time.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels.api as sm

from utils.conditions import (
    add_condition_columns,
    session_sort_key,
)

from utils.colors import get_subject_color
from utils.style import pretty_axes, save_pretty_fig, get_robust_ylims

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

def plot_reaction_time_one_figure_per_participant(
    rt_data,
    plots_dir,
    subject_colors,
    y_col="reaction_time_norm_delta",
    y_label="Reaction time change from first 10 trials (ms)",
    block_size=80,
    fig_prefix="reaction_time_by_isi_over_trials",
    x_col="trial_num",
    x_label="Global trial number",
    xlim=(0.5, 400.5),
    hac_maxlags=10,
    band="sd",
):
    """
    Plot reaction time over the full session.

    One figure per participant:
        - one subplot per ISI / stimulation condition
        - x-axis = global trial number
        - y-axis = reaction-time metric
        - linear regression with confidence interval
        - vertical lines = global block boundaries

    The shaded region represents the confidence interval around the
    predicted mean regression line, not the standard deviation of the data.
    """

    rt_data = rt_data.copy()

    if x_col not in rt_data.columns:
        if x_col == "trial_within_isi":
            rt_data = rt_data.sort_values(
                ["participant_id", "isi_bin", "trial_num"]
            )

            rt_data["trial_within_isi"] = (
                rt_data
                .groupby(["participant_id", "isi_bin"])
                .cumcount()
                + 1
            )
        else:
            raise KeyError(f"{x_col} not found in rt_data.")

    participants = rt_data["participant_id"].dropna().unique()

    isi_values = np.array(
        sorted(rt_data["isi_bin"].dropna().unique()),
        dtype=float,
    )

    for participant_id in participants:

        df_participant = rt_data[
            rt_data["participant_id"] == participant_id
        ].copy()

        if df_participant.empty:
            continue

        color = get_subject_color(
            participant_id,
            subject_colors,
        )

        valid_y = df_participant.loc[
            df_participant["reaction_time_valid"].astype(bool)
            & np.isfinite(df_participant[y_col]),
            y_col,
        ]

        if len(valid_y) > 0:
            ylims = get_robust_ylims(valid_y)
        else:
            ylims = (-1, 1)

        invalid_y = (
            ylims[0]
            + 0.05 * (ylims[1] - ylims[0])
        )

        fig, axes = plt.subplots(
            nrows=len(isi_values),
            ncols=1,
            figsize=(11, 2.8 * len(isi_values)),
            sharex=True,
            sharey=True,
        )

        if len(isi_values) == 1:
            axes = [axes]

        for ax, isi in zip(axes, isi_values):

            df_isi = df_participant[
                df_participant["isi_bin"] == isi
            ].sort_values(x_col)

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

            valid = (
                df_isi["reaction_time_valid"].astype(bool)
                & np.isfinite(df_isi[y_col])
                & np.isfinite(df_isi[x_col])
            )

            invalid = ~df_isi["reaction_time_valid"].astype(bool)

            ax.scatter(
                df_isi.loc[valid, x_col],
                df_isi.loc[valid, y_col],
                color=color,
                edgecolor="black",
                linewidth=0.25,
                s=28,
                alpha=0.65,
                zorder=3,
            )

            if invalid.any():
                ax.scatter(
                    df_isi.loc[invalid, x_col],
                    np.full(invalid.sum(), invalid_y),
                    marker="x",
                    color="red",
                    s=36,
                    alpha=0.9,
                    zorder=4,
                )

            x = df_isi.loc[valid, x_col].to_numpy(dtype=float)
            y = df_isi.loc[valid, y_col].to_numpy(dtype=float)

            title_text = f"ISI {isi:.2f} s"

            regression = fit_linear_regression(
                x=x,
                y=y,
                x_fit=np.linspace(
                    np.nanmin(x),
                    np.nanmax(x),
                    200,
                ) if len(x) >= 3 else None,
                band=band,
                hac_maxlags=hac_maxlags,
            )

            if regression is not None:

                ax.fill_between(
                    regression["x_fit"],
                    regression["band_lower"],
                    regression["band_upper"],
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                    label=f"{band.upper()} band around fitted line",
                    zorder=1,
                )

                ax.plot(
                    regression["x_fit"],
                    regression["y_fit"],
                    color=color,
                    linewidth=2.5,
                    label="Linear regression",
                    zorder=2,
                )

                title_text += (
                    f" | slope={regression['slope']:.3f}"
                    f", r={regression['rvalue']:.3f}"
                    f", p={regression['slope_pvalue']:.3g}"
                )

            if "norm" in y_col:
                ax.axhline(
                    0,
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    zorder=0,
                )

            if block_size is not None and xlim is not None:

                block_boundaries = np.arange(
                    block_size,
                    xlim[1],
                    block_size,
                )

                for boundary in block_boundaries:
                    ax.axvline(
                        boundary + 0.5,
                        color="gray",
                        linestyle=":",
                        linewidth=1,
                        alpha=0.7,
                        zorder=0,
                    )

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylims)
            ax.set_title(
                title_text,
                fontweight="bold",
                fontsize=10,
            )

            pretty_axes(ax)

        axes[-1].set_xlabel(x_label)
        fig.supylabel(y_label)

        handles, labels = axes[0].get_legend_handles_labels()

        if handles:
            unique_legend = dict(zip(labels, handles))

            axes[0].legend(
                unique_legend.values(),
                unique_legend.keys(),
                frameon=False,
            )

        fig.suptitle(
            f"{participant_id} — EMG reaction time over global trials by ISI",
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
    block_size=80,
    fig_prefix="reaction_time_variability_by_isi_over_trials",
    x_col="trial_num",
    x_label="Global trial number",
    xlim=(0.5, 400.5),
    hac_maxlags=10,
    band="sd",
):
    """
    Plot rolling reaction-time variability over the full session.

    One figure per participant:
        - one subplot per ISI / stimulation condition
        - x-axis = global trial number
        - y-axis = rolling SD of reaction time
        - linear regression with confidence interval

    Warning
    -------
    Consecutive rolling SD values are strongly dependent because their
    windows overlap. HAC robust standard errors are therefore used, but
    this analysis remains exploratory.
    """

    rt_data = rt_data.copy()

    if x_col not in rt_data.columns:
        if x_col == "trial_within_isi":
            rt_data = rt_data.sort_values(
                ["participant_id", "isi_bin", "trial_num"]
            )

            rt_data["trial_within_isi"] = (
                rt_data
                .groupby(["participant_id", "isi_bin"])
                .cumcount()
                + 1
            )
        else:
            raise KeyError(f"{x_col} not found in rt_data.")

    participants = rt_data["participant_id"].dropna().unique()

    isi_values = np.array(
        sorted(rt_data["isi_bin"].dropna().unique()),
        dtype=float,
    )

    variability_col = "reaction_time_rolling_sd"
    rt_data[variability_col] = np.nan

    for _, idx in rt_data.groupby(
        ["participant_id", "isi_bin"]
    ).groups.items():

        df_group = (
            rt_data
            .loc[idx]
            .sort_values(x_col)
            .copy()
        )

        valid_rt = df_group[rt_col].where(
            df_group["reaction_time_valid"].astype(bool)
        )

        rolling_sd = valid_rt.rolling(
            window=rolling_window,
            min_periods=min_periods,
            center=True,
        ).std()

        rt_data.loc[
            df_group.index,
            variability_col,
        ] = rolling_sd

    for participant_id in participants:

        df_participant = rt_data[
            rt_data["participant_id"] == participant_id
        ].copy()

        if df_participant.empty:
            continue

        color = get_subject_color(
            participant_id,
            subject_colors,
        )

        valid_y = df_participant[variability_col]
        valid_y = valid_y[np.isfinite(valid_y)]

        if len(valid_y) > 0:
            ylims = get_robust_ylims(valid_y)
            ylims = (max(0, ylims[0]), ylims[1])
        else:
            ylims = (0, 1)

        invalid_y = (
            ylims[0]
            + 0.05 * (ylims[1] - ylims[0])
        )

        fig, axes = plt.subplots(
            nrows=len(isi_values),
            ncols=1,
            figsize=(11, 2.8 * len(isi_values)),
            sharex=True,
            sharey=True,
        )

        if len(isi_values) == 1:
            axes = [axes]

        for ax, isi in zip(axes, isi_values):

            df_isi = df_participant[
                df_participant["isi_bin"] == isi
            ].sort_values(x_col)

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

            valid = (
                np.isfinite(df_isi[variability_col])
                & np.isfinite(df_isi[x_col])
            )

            ax.plot(
                df_isi.loc[valid, x_col],
                df_isi.loc[valid, variability_col],
                color=color,
                linewidth=1.5,
                alpha=0.55,
                zorder=2,
            )

            ax.scatter(
                df_isi.loc[valid, x_col],
                df_isi.loc[valid, variability_col],
                color=color,
                edgecolor="black",
                linewidth=0.25,
                s=22,
                alpha=0.55,
                zorder=3,
            )

            invalid = ~df_isi["reaction_time_valid"].astype(bool)

            if invalid.any():
                ax.scatter(
                    df_isi.loc[invalid, x_col],
                    np.full(invalid.sum(), invalid_y),
                    marker="x",
                    color="red",
                    s=36,
                    alpha=0.9,
                    zorder=4,
                )

            x = df_isi.loc[valid, x_col].to_numpy(dtype=float)
            y = df_isi.loc[valid, variability_col].to_numpy(dtype=float)

            title_text = f"ISI {isi:.2f} s"

            regression = fit_linear_regression(
                x=x,
                y=y,
                x_fit=np.linspace(
                    np.nanmin(x),
                    np.nanmax(x),
                    200,
                ) if len(x) >= 3 else None,
                    band=band,
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
                    label=f"{band.upper()} band around fitted line",
                    zorder=1,
                )

                ax.plot(
                    regression["x_fit"],
                    regression["y_fit"],
                    color=color,
                    linewidth=2.5,
                    label="Linear regression",
                    zorder=3,
                )

                title_text += (
                    f" | slope={regression['slope']:.3f}"
                    f", r={regression['rvalue']:.3f}"
                    f", p={regression['slope_pvalue']:.3g}"
                )

            if block_size is not None and xlim is not None:

                block_boundaries = np.arange(
                    block_size,
                    xlim[1],
                    block_size,
                )

                for boundary in block_boundaries:
                    ax.axvline(
                        boundary + 0.5,
                        color="gray",
                        linestyle=":",
                        linewidth=1,
                        alpha=0.7,
                        zorder=0,
                    )

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylims)

            ax.set_title(
                title_text,
                fontweight="bold",
                fontsize=10,
            )

            pretty_axes(ax)

        axes[-1].set_xlabel(x_label)

        fig.supylabel(
            f"Rolling RT variability, {band.upper()} over "
            f"{rolling_window} observations (ms)"
        )

        handles, labels = axes[0].get_legend_handles_labels()

        if handles:
            unique_legend = dict(zip(labels, handles))

            axes[0].legend(
                unique_legend.values(),
                unique_legend.keys(),
                frameon=False,
            )

        fig.suptitle(
            f"{participant_id} — reaction-time variability "
            f"over global trials by ISI",
            fontweight="bold",
            y=1.01,
        )

        fig_name = f"{fig_prefix}_{participant_id}.png"

        save_pretty_fig(
            fig,
            fig_name,
            plots_dir,
        )
        
def quick_plot_rt_detection(
    signal_data,
    timing_data,
    rt_data,
    participant_id,
    trial_num,
    emg_var,
    baseline_window_s=(-0.5, -0.1),
    response_window_s=(0.0, 1.5),
):
    signal_p = signal_data[
        signal_data["participant_id"] == participant_id
    ].sort_values("timestamp")

    timing_p = timing_data[
        timing_data["participant_id"] == participant_id
    ].sort_values("trial_num")

    rt_p = rt_data[
        (rt_data["participant_id"] == participant_id)
        & (rt_data["trial_num"] == trial_num)
    ]

    event_time = timing_p.loc[
        timing_p["trial_num"] == trial_num,
        "event"
    ].iloc[0]

    t = signal_p["timestamp"].to_numpy(dtype=float)
    y = signal_p[emg_var].to_numpy(dtype=float)

    plot_start = event_time + baseline_window_s[0] - 0.2
    plot_end = event_time + response_window_s[1]

    idx = (t >= plot_start) & (t <= plot_end)

    plt.figure(figsize=(11, 4))
    plt.plot(t[idx] - event_time, y[idx], linewidth=1.5)

    plt.axvline(0, color="black", linestyle="--", label="Event")
    plt.axvspan(
        baseline_window_s[0],
        baseline_window_s[1],
        color="gray",
        alpha=0.25,
        label="Baseline window",
    )

    if not rt_p.empty:
        threshold = rt_p["emg_onset_threshold"].iloc[0]
        onset_time = rt_p["emg_onset_time"].iloc[0]
        is_valid = rt_p["reaction_time_valid"].iloc[0]

        plt.axhline(threshold, color="red", linestyle="--", label="Threshold")

        if np.isfinite(onset_time):
            plt.axvline(
                onset_time - event_time,
                color="green",
                linestyle="--",
                label=f"Detected onset | valid={is_valid}",
            )

    plt.xlabel("Time from event (s)")
    plt.ylabel(emg_var)
    plt.title(f"{participant_id} | trial {trial_num}")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def _make_safe_filename(value):
    text = str(value).strip().replace(" ", "_")

    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-", "."}:
            safe.append(ch)
        else:
            safe.append("_")

    return "".join(safe).strip("_")


def _format_combined_regression_legend_label(
    participant_id,
    slope,
    pvalue,
):
    if np.isfinite(pvalue):
        p_text = f"p={pvalue:.3g}"
    else:
        p_text = "p=NA"

    return f"{participant_id} | slope={slope:.3f}, {p_text}"


def _format_band_description(
    band="sd",
    sd_multiplier=1.0,
    ci_level=0.95,
):
    if band == "sd":
        return f"±{sd_multiplier:g} residual SD"

    if band == "ci":
        return f"{ci_level * 100:.0f}% CI"

    return "no band"


def _condition_sort_key(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _format_condition_label(value):
    try:
        return f"ISI {float(value):.2f} s"
    except (TypeError, ValueError):
        return str(value)


def _prepare_rt_x_column(
    rt_data,
    x_col="trial_num",
):
    rt_data = rt_data.copy()

    if x_col in rt_data.columns:
        return rt_data

    if x_col == "trial_within_isi":
        rt_data = rt_data.sort_values(
            ["participant_id", "isi_bin", "trial_num"]
        )

        rt_data["trial_within_isi"] = (
            rt_data
            .groupby(["participant_id", "isi_bin"])
            .cumcount()
            + 1
        )

        return rt_data

    raise KeyError(f"{x_col} not found in rt_data.")


def _y_limits_from_regressions(
    regressions,
    force_lower_zero=False,
):
    values = []

    for item in regressions:
        regression = item["regression"]

        values.append(regression["y_fit"])

        band_lower = regression.get("band_lower", None)
        band_upper = regression.get("band_upper", None)

        if band_lower is not None:
            values.append(band_lower)

        if band_upper is not None:
            values.append(band_upper)

    if not values:
        return (0, 1) if force_lower_zero else (-1, 1)

    values = np.concatenate(values)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return (0, 1) if force_lower_zero else (-1, 1)

    ymin = np.nanmin(values)
    ymax = np.nanmax(values)

    if ymin == ymax:
        pad = 1.0 if ymin == 0 else abs(ymin) * 0.1
    else:
        pad = 0.08 * (ymax - ymin)

    ymin = ymin - pad
    ymax = ymax + pad

    if force_lower_zero:
        ymin = max(0, ymin)

    return ymin, ymax


def _collect_participant_rt_regressions(
    rt_data,
    y_col,
    x_col="trial_num",
    valid_col="reaction_time_valid",
    hac_maxlags=10,
    band="sd",
    sd_multiplier=1.0,
    ci_level=0.95,
    config=None,
):
    regressions = []

    participants = sorted(
        rt_data["participant_id"].dropna().astype(str).unique(),
        key=lambda x: session_sort_key(x, config=config),
    )

    for participant_id in participants:

        dfp = (
            rt_data[
                rt_data["participant_id"] == participant_id
            ]
            .sort_values(x_col)
            .copy()
        )

        if dfp.empty:
            continue

        valid = (
            np.isfinite(dfp[x_col])
            & np.isfinite(dfp[y_col])
        )

        if valid_col is not None and valid_col in dfp.columns:
            valid = valid & dfp[valid_col].astype(bool)

        x = dfp.loc[valid, x_col].to_numpy(dtype=float)
        y = dfp.loc[valid, y_col].to_numpy(dtype=float)

        if len(x) < 3:
            continue

        regression = fit_linear_regression(
            x=x,
            y=y,
            x_fit=np.linspace(
                np.nanmin(x),
                np.nanmax(x),
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

    return regressions


def _plot_combined_rt_regressions(
    regressions,
    plots_dir,
    subject_colors,
    filename,
    title,
    y_label,
    x_label="Trial number",
    xlim=(0.5, 400.5),
    block_size=80,
    alpha=0.05,
    band="sd",
    sd_multiplier=1.0,
    ci_level=0.95,
    show_zero_line=False,
    force_lower_zero=False,
):
    if not regressions:
        print(f"No regressions available for {filename}.")
        return

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
            str(d["participant_id"]),
        ),
    )

    fig, ax = plt.subplots(
        figsize=(12.5, 7),
    )

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

    ymin, ymax = _y_limits_from_regressions(
        regressions,
        force_lower_zero=force_lower_zero,
    )

    ax.set_ylim(ymin, ymax)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if show_zero_line and ymin <= 0 <= ymax:
        ax.axhline(
            0,
            color="black",
            linestyle=":",
            linewidth=1.2,
            alpha=0.6,
            zorder=0,
        )

    if block_size is not None and xlim is not None:
        block_boundaries = np.arange(
            block_size,
            xlim[1],
            block_size,
        )

        for boundary in block_boundaries:
            ax.axvline(
                boundary + 0.5,
                color="gray",
                linestyle=":",
                linewidth=1,
                alpha=0.55,
                zorder=0,
            )

    ax.set_xlabel(x_label)
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

    ax.legend(
        title="Participant-level HAC regressions",
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

def plot_reaction_time_regressions_combined(
    rt_data,
    plots_dir,
    subject_colors,
    y_col="reaction_time_norm_delta",
    y_label="Reaction time change from first 10 trials (ms)",
    x_col="trial_num",
    x_label="Global trial number",
    xlim=(0.5, 400.5),
    block_size=80,
    fig_prefix="reaction_time_regressions_combined",
    alpha=0.05,
    hac_maxlags=10,
    band="sd",
    sd_multiplier=1.0,
    ci_level=0.95,
    split_by_isi=False,
    config=None,
):
    """
    Combined participant-level regression plot for reaction time.

    If split_by_isi=False:
        one figure with all trials merged per participant.

    If split_by_isi=True:
        one combined figure per ISI / stimulation condition.
    """
    

    rt_data = _prepare_rt_x_column(
        rt_data,
        x_col=x_col,
    )

    if split_by_isi:
        isi_values = sorted(
            rt_data["isi_bin"].dropna().unique(),
            key=_condition_sort_key,
        )

        for isi in isi_values:
            df_condition = rt_data[
                rt_data["isi_bin"] == isi
            ].copy()

            condition_label = _format_condition_label(isi)
            condition_safe = _make_safe_filename(condition_label)

            regressions = _collect_participant_rt_regressions(
                rt_data=df_condition,
                y_col=y_col,
                x_col=x_col,
                valid_col="reaction_time_valid",
                hac_maxlags=hac_maxlags,
                band=band,
                sd_multiplier=sd_multiplier,
                ci_level=ci_level,
            )

            _plot_combined_rt_regressions(
                regressions=regressions,
                plots_dir=plots_dir,
                subject_colors=subject_colors,
                filename=f"{fig_prefix}_{condition_safe}.png",
                title=(
                    "Participant regression lines for EMG reaction time — "
                    f"{condition_label}"
                ),
                y_label=y_label,
                x_label=x_label,
                xlim=xlim,
                block_size=block_size,
                alpha=alpha,
                band=band,
                sd_multiplier=sd_multiplier,
                ci_level=ci_level,
                show_zero_line=("norm" in y_col),
                force_lower_zero=False,
            )

        return

    regressions = _collect_participant_rt_regressions(
        rt_data=rt_data,
        y_col=y_col,
        x_col=x_col,
        valid_col="reaction_time_valid",
        hac_maxlags=hac_maxlags,
        band=band,
        sd_multiplier=sd_multiplier,
        ci_level=ci_level,
    )

    _plot_combined_rt_regressions(
        regressions=regressions,
        plots_dir=plots_dir,
        subject_colors=subject_colors,
        filename=f"{fig_prefix}.png",
        title="Participant regression lines for EMG reaction time",
        y_label=y_label,
        x_label=x_label,
        xlim=xlim,
        block_size=block_size,
        alpha=alpha,
        band=band,
        sd_multiplier=sd_multiplier,
        ci_level=ci_level,
        show_zero_line=("norm" in y_col),
        force_lower_zero=False,
    )

def add_reaction_time_rolling_variability(
    rt_data,
    rt_col="reaction_time_ms",
    rolling_window=5,
    min_periods=5,
    x_col="trial_num",
    group_cols=("participant_id",),
    variability_col="reaction_time_rolling_sd",
):
    """
    Add rolling RT variability to rt_data.

    By default, variability is computed across all trials per participant.
    If you want ISI-specific variability, use:

        group_cols=("participant_id", "isi_bin")
    """

    rt_data = _prepare_rt_x_column(
        rt_data,
        x_col=x_col,
    )

    rt_data[variability_col] = np.nan

    for _, idx in rt_data.groupby(
        list(group_cols),
        sort=False,
    ).groups.items():

        df_group = (
            rt_data
            .loc[idx]
            .sort_values(x_col)
            .copy()
        )

        if "reaction_time_valid" in df_group.columns:
            valid_rt = df_group[rt_col].where(
                df_group["reaction_time_valid"].astype(bool)
            )
        else:
            valid_rt = df_group[rt_col]

        rolling_sd = valid_rt.rolling(
            window=rolling_window,
            min_periods=min_periods,
            center=True,
        ).std()

        rt_data.loc[
            df_group.index,
            variability_col,
        ] = rolling_sd

    return rt_data


def plot_reaction_time_variability_regressions_combined(
    rt_data,
    plots_dir,
    subject_colors,
    rt_col="reaction_time_ms",
    rolling_window=5,
    min_periods=5,
    x_col="trial_num",
    x_label="Global trial number",
    xlim=(0.5, 400.5),
    block_size=80,
    fig_prefix="reaction_time_variability_regressions_combined",
    alpha=0.05,
    hac_maxlags=10,
    band="sd",
    sd_multiplier=1.0,
    ci_level=0.95,
    split_by_isi=False,
    variability_group_by_isi=None,
):
    """
    Combined participant-level regression plot for rolling RT variability.

    If split_by_isi=False:
        one figure with variability computed across all trials per participant.

    If split_by_isi=True:
        one figure per ISI / stimulation condition.

    variability_group_by_isi controls how the rolling SD is computed.
    If None, it follows split_by_isi.
    """

    if variability_group_by_isi is None:
        variability_group_by_isi = split_by_isi

    variability_col = "reaction_time_rolling_sd"

    if variability_group_by_isi:
        group_cols = ("participant_id", "isi_bin")
    else:
        group_cols = ("participant_id",)

    rt_data = add_reaction_time_rolling_variability(
        rt_data=rt_data,
        rt_col=rt_col,
        rolling_window=rolling_window,
        min_periods=min_periods,
        x_col=x_col,
        group_cols=group_cols,
        variability_col=variability_col,
    )

    y_label = (
        f"Rolling RT variability "
        f"SD over {rolling_window} trials (ms)"
    )

    if split_by_isi:
        isi_values = sorted(
            rt_data["isi_bin"].dropna().unique(),
            key=_condition_sort_key,
        )

        for isi in isi_values:
            df_condition = rt_data[
                rt_data["isi_bin"] == isi
            ].copy()

            condition_label = _format_condition_label(isi)
            condition_safe = _make_safe_filename(condition_label)

            regressions = _collect_participant_rt_regressions(
                rt_data=df_condition,
                y_col=variability_col,
                x_col=x_col,
                valid_col=None,
                hac_maxlags=hac_maxlags,
                band=band,
                sd_multiplier=sd_multiplier,
                ci_level=ci_level,
            )

            _plot_combined_rt_regressions(
                regressions=regressions,
                plots_dir=plots_dir,
                subject_colors=subject_colors,
                filename=f"{fig_prefix}_{condition_safe}.png",
                title=(
                    "Participant regression lines for RT variability — "
                    f"{condition_label}"
                ),
                y_label=y_label,
                x_label=x_label,
                xlim=xlim,
                block_size=block_size,
                alpha=alpha,
                band=band,
                sd_multiplier=sd_multiplier,
                ci_level=ci_level,
                show_zero_line=False,
                force_lower_zero=True,
            )

        return

    regressions = _collect_participant_rt_regressions(
        rt_data=rt_data,
        y_col=variability_col,
        x_col=x_col,
        valid_col=None,
        hac_maxlags=hac_maxlags,
        band=band,
        sd_multiplier=sd_multiplier,
        ci_level=ci_level,
    )

    _plot_combined_rt_regressions(
        regressions=regressions,
        plots_dir=plots_dir,
        subject_colors=subject_colors,
        filename=f"{fig_prefix}.png",
        title="Participant regression lines for RT variability",
        y_label=y_label,
        x_label=x_label,
        xlim=xlim,
        block_size=block_size,
        alpha=alpha,
        band=band,
        sd_multiplier=sd_multiplier,
        ci_level=ci_level,
        show_zero_line=False,
        force_lower_zero=True,
    )