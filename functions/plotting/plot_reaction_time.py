import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels.api as sm

from utils.colors import get_subject_color
from utils.style import pretty_axes, save_pretty_fig, get_robust_ylims

def fit_linear_regression_with_ci(
    x,
    y,
    x_fit=None,
    ci_level=0.95,
    hac_maxlags=None,
):
    """
    Fit a linear regression and compute a confidence interval around the
    predicted mean regression line.

    Parameters
    ----------
    x, y
        Observed values.

    x_fit
        X values at which predictions are calculated. If None, 200 values
        spanning the observed x range are used.

    ci_level
        Confidence level for the regression band.

    hac_maxlags
        If provided, use HAC robust standard errors with this number of lags.
        This is useful for temporally ordered trial data.

    Returns
    -------
    dict containing:
        x_fit
        y_fit
        ci_lower
        ci_upper
        slope
        intercept
        slope_pvalue
        rvalue
        r_squared
    """

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

    X = sm.add_constant(x)

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

    prediction = result.get_prediction(X_fit)

    prediction_frame = prediction.summary_frame(
        alpha=1 - ci_level,
    )

    # r is kept for consistency with the previous figures.
    simple_regression = linregress(x, y)

    return {
        "x_fit": x_fit,
        "y_fit": prediction_frame["mean"].to_numpy(),
        "ci_lower": prediction_frame["mean_ci_lower"].to_numpy(),
        "ci_upper": prediction_frame["mean_ci_upper"].to_numpy(),
        "intercept": result.params[0],
        "slope": result.params[1],
        "slope_pvalue": result.pvalues[1],
        "rvalue": simple_regression.rvalue,
        "r_squared": result.rsquared,
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
    ci_level=0.95,
    hac_maxlags=10,
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

            regression = fit_linear_regression_with_ci(
                x=x,
                y=y,
                x_fit=np.linspace(
                    np.nanmin(x),
                    np.nanmax(x),
                    200,
                ) if len(x) >= 3 else None,
                ci_level=ci_level,
                hac_maxlags=hac_maxlags,
            )

            if regression is not None:

                ax.fill_between(
                    regression["x_fit"],
                    regression["ci_lower"],
                    regression["ci_upper"],
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                    label=f"{int(ci_level * 100)}% CI",
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
    ci_level=0.95,
    hac_maxlags=10,
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

            regression = fit_linear_regression_with_ci(
                x=x,
                y=y,
                x_fit=np.linspace(
                    np.nanmin(x),
                    np.nanmax(x),
                    200,
                ) if len(x) >= 3 else None,
                ci_level=ci_level,
                hac_maxlags=hac_maxlags,
            )

            if regression is not None:

                ax.fill_between(
                    regression["x_fit"],
                    regression["ci_lower"],
                    regression["ci_upper"],
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                    label=f"{int(ci_level * 100)}% CI",
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
            f"Rolling RT variability, SD over "
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
