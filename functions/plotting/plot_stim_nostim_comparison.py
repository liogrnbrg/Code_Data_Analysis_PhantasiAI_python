# plot_stim_nostim_comparison.py

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from itertools import cycle

from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    DrawingArea,
    HPacker,
    TextArea,
    VPacker,
)
from matplotlib.lines import Line2D

from utils.colors import get_subject_color
from utils.style import pretty_axes, save_pretty_fig

from utils.conditions import (
    add_condition_columns,
    session_sort_key,
)


def _make_safe_filename(value):
    text = str(value).strip().replace(" ", "_")

    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-", "."}:
            safe.append(ch)
        else:
            safe.append("_")

    return "".join(safe).strip("_")


def _pvalue_to_stars(pvalue):
    if not np.isfinite(pvalue):
        return "NA"
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def _condition_sort_key(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _format_isi_label(value):
    try:
        return f"ISI {float(value):.2f} s"
    except (TypeError, ValueError):
        return str(value)


def _ensure_condition_columns(data, config=None):
    data = data.copy()

    required = {
        "base_participant",
        "condition",
        "condition_label",
        "session_number",
    }

    if not required.issubset(data.columns):
        data = add_condition_columns(
            data,
            config=config,
            participant_col="participant_id",
        )

    return data


def _session_label(session_id, base_participant):
    """
    Examples:
        Lio_NOSTIM -> NOSTIM
        Lio_FIXED_STIM -> FIXED STIM
        Lio_STIM_2 -> STIM 2
    """

    session_id = str(session_id)
    prefix = f"{base_participant}_"

    if session_id.startswith(prefix):
        return session_id[len(prefix):].replace("_", " ")

    return session_id.replace("_", " ")


def _session_marker(condition):
    markers = {
        "NOSTIM": "s",
        "FIXED_STIM": "D",
        "STIM": "o",
        "UNKNOWN": "x",
    }

    return markers.get(str(condition), "o")


def _robust_ylims(
    values,
    lower_pct=1,
    upper_pct=99,
    pad_fraction=0.20,
    force_lower_zero=False,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return (0, 1) if force_lower_zero else (-1, 1)

    lower = np.nanpercentile(values, lower_pct)
    upper = np.nanpercentile(values, upper_pct)

    padding = pad_fraction * (upper - lower)

    if not np.isfinite(padding) or padding == 0:
        padding = 1.0

    ymin = lower - padding
    ymax = upper + padding

    if force_lower_zero:
        ymin = max(0, ymin)

    return ymin, ymax


def _add_colored_regression_box(
    ax,
    regression_entries,
    loc="upper right",
    fontsize=8.5,
):
    """
    Add one combined legend/statistics box.

    Each row contains:
        colored regression-line swatch + session label + regression stats
    """

    if not regression_entries:
        return

    rows = []

    for entry in regression_entries:
        color = entry["color"]
        label = entry["label"]
        stats_text = entry["stats_text"]

        color_swatch = DrawingArea(
            18,
            8,
            0,
            0,
        )

        color_swatch.add_artist(
            Line2D(
                [0, 18],
                [4, 4],
                color=color,
                linewidth=3.2,
            )
        )

        text = TextArea(
            f"{label}: {stats_text}",
            textprops={
                "fontsize": fontsize,
                "color": "black",
            },
        )

        row = HPacker(
            children=[
                color_swatch,
                text,
            ],
            align="center",
            pad=0,
            sep=4,
        )

        rows.append(row)

    packed_rows = VPacker(
        children=rows,
        align="right",
        pad=0,
        sep=1.5,
    )

    anchored_box = AnchoredOffsetbox(
        loc=loc,
        child=packed_rows,
        pad=0.25,
        borderpad=0.45,
        frameon=True,
    )

    anchored_box.patch.set_facecolor("white")
    anchored_box.patch.set_edgecolor("gray")
    anchored_box.patch.set_alpha(0.88)
    anchored_box.patch.set_linewidth(0.8)

    anchored_box.set_zorder(10)

    ax.add_artist(anchored_box)


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
    Fit a linear regression and optionally calculate a shaded band.
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
        x_fit = np.asarray(
            x_fit,
            dtype=float,
        )

    X = sm.add_constant(
        x,
        has_constant="add",
    )

    model = sm.OLS(
        y,
        X,
    )

    if hac_maxlags is None:
        result = model.fit()
    else:
        result = model.fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": hac_maxlags,
            },
        )

    X_fit = sm.add_constant(
        x_fit,
        has_constant="add",
    )

    y_fit = np.asarray(
        result.predict(X_fit),
        dtype=float,
    )

    fitted_observed = np.asarray(
        result.predict(X),
        dtype=float,
    )

    residuals = y - fitted_observed

    if result.df_resid > 0:
        residual_sd = np.sqrt(
            np.sum(residuals ** 2)
            / result.df_resid
        )
    else:
        residual_sd = np.nan

    if band == "sd":
        band_lower = (
            y_fit
            - sd_multiplier * residual_sd
        )

        band_upper = (
            y_fit
            + sd_multiplier * residual_sd
        )

    elif band == "ci":
        prediction = result.get_prediction(
            X_fit
        )

        prediction_frame = (
            prediction.summary_frame(
                alpha=1 - ci_level,
            )
        )

        band_lower = prediction_frame[
            "mean_ci_lower"
        ].to_numpy(dtype=float)

        band_upper = prediction_frame[
            "mean_ci_upper"
        ].to_numpy(dtype=float)

    else:
        band_lower = None
        band_upper = None

    correlation = np.corrcoef(
        x,
        y,
    )[0, 1]

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


def _plot_condition_overlay(
    long_data,
    plots_dir,
    subject_colors,
    y_col,
    y_label,
    fig_prefix,
    title_suffix,
    valid_col=None,
    block_size=80,
    xlim=(0.5, 400.5),
    band="sd",
    show_individual_points=True,
    force_lower_zero=False,
    show_zero_line=True,
    config=None,
):
    """
    Generic condition overlay.

    One figure per base participant.
    One panel per ISI.
    One line per recording session.
    """

    long_data = _ensure_condition_columns(
        long_data,
        config=config,
    )

    required_cols = {
        "base_participant",
        "participant_id",
        "condition",
        "condition_label",
        "isi_bin",
        "trial_num",
        y_col,
    }

    if valid_col is not None:
        required_cols.add(valid_col)

    missing = required_cols.difference(long_data.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    print(f"\nSessions received by {fig_prefix}:")
    print(
        long_data.groupby(
            [
                "base_participant",
                "participant_id",
                "condition",
            ],
            dropna=False,
        ).size()
    )

    base_participants = (
        long_data["base_participant"]
        .dropna()
        .astype(str)
        .unique()
    )

    for base_participant in base_participants:
        participant_data = long_data[
            long_data["base_participant"].astype(str)
            == str(base_participant)
        ].copy()

        session_ids = sorted(
            participant_data["participant_id"]
            .dropna()
            .astype(str)
            .unique(),
            key=lambda x: session_sort_key(
                x,
                config=config,
            ),
        )

        if len(session_ids) == 0:
            continue

        isi_values = sorted(
            participant_data["isi_bin"]
            .dropna()
            .unique(),
            key=_condition_sort_key,
        )

        if not isi_values:
            continue

        valid_global = (
            np.isfinite(participant_data[y_col])
        )

        if valid_col is not None:
            valid_global = (
                valid_global
                & participant_data[valid_col].astype(bool)
            )

        ylims = _robust_ylims(
            participant_data.loc[valid_global, y_col],
            force_lower_zero=force_lower_zero,
        )

        fig, axes = plt.subplots(
            nrows=len(isi_values),
            ncols=1,
            figsize=(7.5, max(3.2, 2.9 * len(isi_values))),
            sharex=True,
            sharey=True,
        )

        if len(isi_values) == 1:
            axes = [axes]

        for ax, isi in zip(axes, isi_values):
            isi_data = participant_data[
                participant_data["isi_bin"] == isi
            ].copy()

            regression_entries = []

            for session_id in session_ids:
                session_data = isi_data[
                    isi_data["participant_id"].astype(str)
                    == str(session_id)
                ].copy()

                if session_data.empty:
                    continue

                valid_session = (
                    np.isfinite(session_data[y_col])
                    & np.isfinite(session_data["trial_num"])
                )

                if valid_col is not None:
                    valid_session = (
                        valid_session
                        & session_data[valid_col].astype(bool)
                    )

                session_data = (
                    session_data
                    .loc[valid_session]
                    .sort_values("trial_num")
                    .copy()
                )

                if session_data.empty:
                    continue

                color = get_subject_color(
                    session_id,
                    subject_colors,
                )

                condition = (
                    session_data["condition"]
                    .dropna()
                    .astype(str)
                    .iloc[0]
                )

                label = _session_label(
                    session_id,
                    base_participant,
                )

                marker = _session_marker(condition)

                if show_individual_points:
                    ax.scatter(
                        session_data["trial_num"],
                        session_data[y_col],
                        color=color,
                        marker=marker,
                        label=label,
                        edgecolor="black",
                        linewidth=0.25,
                        s=27,
                        alpha=0.55,
                        zorder=2,
                    )

                x = session_data[
                    "trial_num"
                ].to_numpy(dtype=float)

                y = session_data[
                    y_col
                ].to_numpy(dtype=float)

                regression = fit_linear_regression(
                    x=x,
                    y=y,
                    x_fit=np.linspace(
                        np.nanmin(x),
                        np.nanmax(x),
                        300,
                    ),
                    band=band,
                    ci_level=0.95,
                    sd_multiplier=1.0,
                    hac_maxlags=10,
                )

                if regression is None:
                    continue

                if (
                    regression["band_lower"] is not None
                    and regression["band_upper"] is not None
                ):
                    ax.fill_between(
                        regression["x_fit"],
                        regression["band_lower"],
                        regression["band_upper"],
                        color=color,
                        alpha=0.18,
                        linewidth=0,
                        zorder=1,
                    )

                ax.plot(
                    regression["x_fit"],
                    regression["y_fit"],
                    color=color,
                    linewidth=2.8,
                    linestyle="-",
                    label=(
                        None
                        if show_individual_points
                        else label
                    ),
                    zorder=3,
                )

                pvalue = regression["slope_pvalue"]
                significance = _pvalue_to_stars(pvalue)

                regression_entries.append(
                    {
                        "color": color,
                        "label": label,
                        "stats_text": (
                            f"slope={regression['slope']:.3f}, "
                            f"p={pvalue:.3g} {significance}, "
                            f"R²={regression['r_squared']:.2f}"
                        ),
                    }
                )

            if regression_entries:
                _add_colored_regression_box(
                    ax=ax,
                    regression_entries=regression_entries,
                    loc="upper right",
                    fontsize=7.5,
                )

            if show_zero_line and ylims[0] <= 0 <= ylims[1]:
                ax.axhline(
                    0,
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.8,
                    zorder=1,
                )

            if block_size is not None and xlim is not None:
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

            if xlim is not None:
                ax.set_xlim(*xlim)

            ax.set_ylim(*ylims)

            ax.set_title(
                _format_isi_label(isi),
                fontweight="bold",
                fontsize=10,
            )

            pretty_axes(ax)

        axes[-1].set_xlabel("Global trial number")
        fig.supylabel(y_label)

        fig.suptitle(
            f"{base_participant} — {title_suffix}",
            fontweight="bold",
            y=0.95,
        )

        save_pretty_fig(
            fig,
            f"{fig_prefix}_{_make_safe_filename(base_participant)}.png",
            plots_dir,
        )


def plot_stim_nostim_rt_overlay(
    long_data,
    trend_stats=None,
    plots_dir=None,
    subject_colors=None,
    y_col="reaction_time_centered_ms",
    y_label="RT change from first 10 trials (ms)",
    block_size=80,
    xlim=(0.5, 400.5),
    fig_prefix="condition_centered_rt",
    band="sd",
    show_individual_points=True,
    config=None,
):
    """
    Generic condition overlay for RT.

    Despite the old function name, this now plots all available conditions:
        NOSTIM
        FIXED_STIM
        STIM

    It does not require all conditions to exist for every participant.
    """

    _ = trend_stats

    _plot_condition_overlay(
        long_data=long_data,
        plots_dir=plots_dir,
        subject_colors=subject_colors,
        y_col=y_col,
        y_label=y_label,
        fig_prefix=fig_prefix,
        title_suffix="RT trends by condition",
        valid_col="reaction_time_valid",
        block_size=block_size,
        xlim=xlim,
        band=band,
        show_individual_points=show_individual_points,
        force_lower_zero=False,
        show_zero_line=True,
        config=config,
    )


def plot_stim_minus_nostim_rt_difference(
    paired_data,
    plots_dir,
    value_col="rt_difference_centered_ms",
    y_label="Test − reference centered RT (ms)",
    block_size=80,
    xlim=(0.5, 400.5),
    fig_prefix="condition_difference_centered_rt",
    band="sd",
    config=None,
):
    """
    Generic paired difference plot.

    Expects paired_data from prepare_rt_stim_nostim_comparison().

    Difference is:
        test condition − reference condition

    Examples:
        STIM − NOSTIM
        FIXED_STIM − NOSTIM
        STIM − FIXED_STIM
    """

    required_cols = {
        "comparison_name",
        "comparison_label",
        "base_participant",
        "test_session_id",
        "reference_session_id",
        "isi_bin",
        "trial_num",
        value_col,
    }

    missing = required_cols.difference(paired_data.columns)

    if missing:
        raise KeyError(
            "This function now expects paired_data from "
            "prepare_rt_stim_nostim_comparison(). "
            f"Missing required columns: {sorted(missing)}"
        )

    if paired_data.empty:
        print("No paired data available for difference plots.")
        return

    group_cols = [
        "comparison_name",
        "comparison_label",
        "base_participant",
        "test_session_id",
        "reference_session_id",
    ]

    for group_values, group in paired_data.groupby(
        group_cols,
        dropna=False,
    ):
        (
            comparison_name,
            comparison_label,
            base_participant,
            test_session_id,
            reference_session_id,
        ) = group_values

        valid = np.isfinite(group[value_col])

        if "pair_valid" in group.columns:
            valid = valid & group["pair_valid"].astype(bool)

        group = group.loc[valid].copy()

        if group.empty:
            print(
                f"No valid paired data for {test_session_id} "
                f"versus {reference_session_id}."
            )
            continue

        isi_values = sorted(
            group["isi_bin"]
            .dropna()
            .unique(),
            key=_condition_sort_key,
        )

        if not isi_values:
            continue

        ylims = _robust_ylims(
            group[value_col],
            force_lower_zero=False,
        )

        fig, axes = plt.subplots(
            nrows=len(isi_values),
            ncols=1,
            figsize=(7.5, max(3.2, 2.9 * len(isi_values))),
            sharex=True,
            sharey=True,
        )

        if len(isi_values) == 1:
            axes = [axes]

        color = get_subject_color(
            test_session_id,
            subject_colors,
        )

        for ax, isi in zip(axes, isi_values):
            isi_data = (
                group[group["isi_bin"] == isi]
                .sort_values("trial_num")
                .copy()
            )

            if isi_data.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No valid paired data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

                ax.set_title(
                    _format_isi_label(isi),
                    fontweight="bold",
                )

                pretty_axes(ax)
                continue

            difference = isi_data[value_col]

            ax.scatter(
                isi_data["trial_num"],
                difference,
                color=color,
                s=27,
                alpha=0.60,
                edgecolor="black",
                linewidth=0.2,
                zorder=2,
            )

            rolling_mean = (
                difference
                .rolling(
                    window=10,
                    min_periods=5,
                    center=True,
                )
                .mean()
            )

            ax.plot(
                isi_data["trial_num"],
                rolling_mean,
                color=color,
                linewidth=2.4,
                zorder=3,
            )

            title = (
                f"{_format_isi_label(isi)}"
                f" | n paired={len(isi_data)}"
            )

            x = isi_data[
                "trial_num"
            ].to_numpy(dtype=float)

            y = difference.to_numpy(dtype=float)

            if len(x) >= 3 and np.nanstd(x) > 0:
                regression = fit_linear_regression(
                    x=x,
                    y=y,
                    x_fit=np.linspace(
                        np.nanmin(x),
                        np.nanmax(x),
                        200,
                    ),
                    band=band,
                    ci_level=0.95,
                    sd_multiplier=1.0,
                    hac_maxlags=10,
                )

                if regression is not None:
                    if (
                        regression["band_lower"] is not None
                        and regression["band_upper"] is not None
                    ):
                        ax.fill_between(
                            regression["x_fit"],
                            regression["band_lower"],
                            regression["band_upper"],
                            color=color,
                            alpha=0.14,
                            linewidth=0,
                            zorder=1,
                        )

                    ax.plot(
                        regression["x_fit"],
                        regression["y_fit"],
                        color="black",
                        linestyle="--",
                        linewidth=1.8,
                        zorder=4,
                    )

                    title += (
                        f" | slope={regression['slope']:.3f}"
                        f" | p={regression['slope_pvalue']:.3g}"
                    )

            ax.axhline(
                0,
                color="black",
                linestyle="-",
                linewidth=1,
            )

            if block_size is not None and xlim is not None:
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
                    )

            if xlim is not None:
                ax.set_xlim(*xlim)

            ax.set_ylim(*ylims)

            ax.set_title(
                title,
                fontweight="bold",
            )

            pretty_axes(ax)

        axes[-1].set_xlabel("Global trial number")
        fig.supylabel(y_label)

        fig.suptitle(
            f"{base_participant} — {comparison_label}\n"
            f"{test_session_id} − {reference_session_id}",
            fontweight="bold",
            y=0.95,
        )

        filename = (
            f"{fig_prefix}_"
            f"{_make_safe_filename(comparison_name)}_"
            f"{_make_safe_filename(test_session_id)}"
            f"_vs_"
            f"{_make_safe_filename(reference_session_id)}.png"
        )

        save_pretty_fig(
            fig,
            filename,
            plots_dir,
        )


def plot_stim_nostim_rt_variability(
    long_data,
    variability_trend_stats=None,
    plots_dir=None,
    subject_colors=None,
    variability_col="reaction_time_rolling_sd_ms",
    block_size=80,
    xlim=(0.5, 400.5),
    fig_prefix="condition_rt_variability",
    band="sd",
    show_individual_points=True,
    config=None,
):
    """
    Generic condition overlay for RT variability.

    Despite the old function name, this now plots all available conditions:
        NOSTIM
        FIXED_STIM
        STIM
    """

    _ = variability_trend_stats

    _plot_condition_overlay(
        long_data=long_data,
        plots_dir=plots_dir,
        subject_colors=subject_colors,
        y_col=variability_col,
        y_label="Rolling RT variability, SD (ms)",
        fig_prefix=fig_prefix,
        title_suffix="RT variability by condition",
        valid_col=None,
        block_size=block_size,
        xlim=xlim,
        band=band,
        show_individual_points=show_individual_points,
        force_lower_zero=True,
        show_zero_line=False,
        config=config,
    )