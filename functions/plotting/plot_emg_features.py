from __future__ import annotations
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from utils.colors import get_subject_color
from utils.style import pretty_axes, save_pretty_fig
from utils.paths import make_safe_filename


def _session_label(session_id, base_participant):
    prefix = f"{base_participant}_"

    if str(session_id).startswith(prefix):
        return str(session_id)[len(prefix):].replace("_", " ")

    return str(session_id).replace("_", " ")


def _get_session_ids(participant_data):
    session_ids = sorted(
        participant_data["participant_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    nostim_sessions = [
        s for s in session_ids
        if "_NOSTIM" in s
    ]

    stim_sessions = [
        s for s in session_ids
        if "_STIM" in s
        and "_NOSTIM" not in s
    ]

    return nostim_sessions, stim_sessions


def fit_linear_regression(
    x,
    y,
    x_fit=None,
    band="sd",
    ci_level=0.95,
    sd_multiplier=1.0,
    hac_maxlags=10,
):
    if band not in {"sd", "ci", None}:
        raise ValueError("band must be 'sd', 'ci', or None.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3 or np.nanstd(x) == 0:
        return None

    if x_fit is None:
        x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    else:
        x_fit = np.asarray(x_fit, dtype=float)

    X = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, X)

    if hac_maxlags is None:
        result = model.fit()
    else:
        result = model.fit(
            cov_type="HAC",
            cov_kwds={"maxlags": hac_maxlags},
        )

    X_fit = sm.add_constant(x_fit, has_constant="add")
    y_fit = np.asarray(result.predict(X_fit), dtype=float)

    residuals = y - np.asarray(result.predict(X), dtype=float)

    if result.df_resid > 0:
        residual_sd = np.sqrt(np.sum(residuals ** 2) / result.df_resid)
    else:
        residual_sd = np.nan

    if band == "sd":
        band_lower = y_fit - sd_multiplier * residual_sd
        band_upper = y_fit + sd_multiplier * residual_sd

    elif band == "ci":
        prediction = result.get_prediction(X_fit)
        frame = prediction.summary_frame(alpha=1 - ci_level)

        band_lower = frame["mean_ci_lower"].to_numpy(dtype=float)
        band_upper = frame["mean_ci_upper"].to_numpy(dtype=float)

    else:
        band_lower = None
        band_upper = None

    return {
        "x_fit": x_fit,
        "y_fit": y_fit,
        "band_lower": band_lower,
        "band_upper": band_upper,
        "slope": result.params[1],
        "slope_pvalue": result.pvalues[1],
        "r_squared": result.rsquared,
        "residual_sd": residual_sd,
    }


def _get_robust_ylims(values, lower_pct=1, upper_pct=99, pad_fraction=0.20):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return -1, 1

    lower = np.nanpercentile(values, lower_pct)
    upper = np.nanpercentile(values, upper_pct)

    pad = pad_fraction * (upper - lower)

    if not np.isfinite(pad) or pad == 0:
        pad = 1.0

    return lower - pad, upper + pad


def plot_emg_feature_overlay(
    features,
    plots_dir,
    subject_colors,
    feature_col,
    y_label,
    fig_prefix=None,
    block_size=40,
    xlim=(0.5, 400.5),
    band="sd",
    show_individual_points=True,
    facet_by_isi=False,
):
    """
    Overlay NOSTIM and all available STIM sessions for a given feature.

    If facet_by_isi=False:
        one panel per base participant, all trials together.

    If facet_by_isi=True:
        one panel per ISI for each base participant.
    """

    required_cols = {
        "participant_id",
        "base_participant",
        "trial_num",
        feature_col,
    }

    if facet_by_isi:
        required_cols.add("isi_bin")

    missing = required_cols.difference(features.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    if fig_prefix is None:
        fig_prefix = f"emg_feature_overlay_{make_safe_filename(feature_col)}"

    marker_cycle = cycle(["o", "^", "D", "v", "P", "X"])

    for base_participant in features["base_participant"].dropna().unique():
        participant_data = features[
            features["base_participant"] == base_participant
        ].copy()

        nostim_sessions, stim_sessions = _get_session_ids(participant_data)
        session_ids = nostim_sessions + stim_sessions

        if len(session_ids) == 0:
            continue

        if facet_by_isi:
            panels = sorted(participant_data["isi_bin"].dropna().unique())
            panel_col = "isi_bin"
        else:
            panels = ["All trials"]
            panel_col = None

        if len(panels) == 0:
            continue

        valid_y = participant_data[feature_col].to_numpy(dtype=float)
        ylims = _get_robust_ylims(valid_y)

        fig, axes = plt.subplots(
            nrows=len(panels),
            ncols=1,
            figsize=(8, max(3.2, 2.8 * len(panels))),
            sharex=True,
            sharey=True,
        )

        if len(panels) == 1:
            axes = [axes]

        session_markers = {}

        for session_id in nostim_sessions:
            session_markers[session_id] = "s"

        for session_id in stim_sessions:
            session_markers[session_id] = next(marker_cycle)

        for ax, panel_value in zip(axes, panels):
            if facet_by_isi:
                panel_data = participant_data[
                    participant_data[panel_col] == panel_value
                ].copy()

                title = f"ISI {float(panel_value):.2f} s"
            else:
                panel_data = participant_data.copy()
                title = "All ISIs combined"

            for session_id in session_ids:
                session_data = panel_data[
                    (panel_data["participant_id"] == session_id)
                    & np.isfinite(panel_data[feature_col])
                    & np.isfinite(panel_data["trial_num"])
                ].sort_values("trial_num")

                if session_data.empty:
                    continue

                color = get_subject_color(session_id, subject_colors)

                label = _session_label(
                    session_id=session_id,
                    base_participant=base_participant,
                )

                if show_individual_points:
                    ax.scatter(
                        session_data["trial_num"],
                        session_data[feature_col],
                        color=color,
                        marker=session_markers[session_id],
                        edgecolor="black",
                        linewidth=0.25,
                        s=28,
                        alpha=0.55,
                        label=label,
                        zorder=2,
                    )

                x = session_data["trial_num"].to_numpy(dtype=float)
                y = session_data[feature_col].to_numpy(dtype=float)

                regression = fit_linear_regression(
                    x=x,
                    y=y,
                    x_fit=np.linspace(
                        np.nanmin(x),
                        np.nanmax(x),
                        250,
                    ),
                    band=band,
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
                        alpha=0.15,
                        linewidth=0,
                        zorder=1,
                    )

                pvalue = regression["slope_pvalue"]

                if pvalue < 0.001:
                    stars = "***"
                elif pvalue < 0.01:
                    stars = "**"
                elif pvalue < 0.05:
                    stars = "*"
                else:
                    stars = "ns"

                ax.plot(
                    regression["x_fit"],
                    regression["y_fit"],
                    color=color,
                    linewidth=2.6,
                    label=(
                        f"{label}: slope={regression['slope']:.4g}, "
                        f"p={pvalue:.3g} {stars}, "
                        f"R²={regression['r_squared']:.2f}"
                    ),
                    zorder=3,
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
                for boundary in range(block_size, int(xlim[1]), block_size):
                    ax.axvline(
                        boundary + 0.5,
                        color="gray",
                        linestyle=":",
                        linewidth=1,
                        alpha=0.65,
                        zorder=1,
                    )

            ax.set_title(
                title,
                fontweight="bold",
                fontsize=10,
            )

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylims)

            pretty_axes(ax)

        axes[-1].set_xlabel("Trial number")
        fig.supylabel(y_label)

        fig.suptitle(
            f"{base_participant} — EMG feature trend",
            fontweight="bold",
            y=0.98,
        )

        handles, labels = axes[0].get_legend_handles_labels()

        if handles:
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
                fontsize=8,
            )

            fig.subplots_adjust(right=0.68)

        safe_participant = make_safe_filename(str(base_participant))

        save_pretty_fig(
            fig,
            f"{fig_prefix}_{safe_participant}.png",
            plots_dir,
        )


def plot_emg_feature_block_summary(
    features,
    plots_dir,
    subject_colors,
    feature_col,
    y_label,
    fig_prefix=None,
):
    """
    Plot block-level means for each session.
    Useful to show learning/fatigue more cleanly than 400 points.
    """

    required_cols = {
        "participant_id",
        "base_participant",
        "trial_block",
        feature_col,
    }

    missing = required_cols.difference(features.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    if fig_prefix is None:
        fig_prefix = f"emg_feature_block_summary_{make_safe_filename(feature_col)}"

    for base_participant in features["base_participant"].dropna().unique():
        participant_data = features[
            features["base_participant"] == base_participant
        ].copy()

        session_ids = sorted(
            participant_data["participant_id"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(session_ids) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 4.8))

        all_y = []

        for session_id in session_ids:
            session_data = participant_data[
                participant_data["participant_id"] == session_id
            ].copy()

            summary = (
                session_data
                .groupby("trial_block", as_index=False)
                .agg(
                    mean_value=(feature_col, "mean"),
                    sem_value=(feature_col, lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.isfinite(x).sum())),
                    n=(feature_col, lambda x: np.isfinite(x).sum()),
                )
            )

            summary = summary[
                np.isfinite(summary["mean_value"])
            ]

            if summary.empty:
                continue

            color = get_subject_color(session_id, subject_colors)

            label = _session_label(
                session_id=session_id,
                base_participant=base_participant,
            )

            ax.errorbar(
                summary["trial_block"],
                summary["mean_value"],
                yerr=summary["sem_value"],
                color=color,
                marker="o",
                linewidth=2.4,
                markersize=6,
                capsize=3,
                label=label,
            )

            all_y.extend(summary["mean_value"].to_numpy(dtype=float))

        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
        )

        ax.set_xlabel("Trial block")
        ax.set_ylabel(y_label)
        ax.set_title(
            f"{base_participant} — EMG feature by trial block",
            fontweight="bold",
        )

        if len(all_y) > 0:
            ax.set_ylim(*_get_robust_ylims(all_y))

        pretty_axes(ax)

        ax.legend(
            frameon=False,
            loc="best",
        )

        safe_participant = make_safe_filename(str(base_participant))

        save_pretty_fig(
            fig,
            f"{fig_prefix}_{safe_participant}.png",
            plots_dir,
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


def _prepare_emg_feature_x_column(
    features,
    x_col="trial_num",
):
    features = features.copy()

    if x_col in features.columns:
        return features

    if x_col == "trial_within_isi":
        features = features.sort_values(
            ["participant_id", "isi_bin", "trial_num"]
        )

        features["trial_within_isi"] = (
            features
            .groupby(["participant_id", "isi_bin"])
            .cumcount()
            + 1
        )

        return features

    raise KeyError(f"{x_col} not found in features.")


def fit_linear_regression(
    x,
    y,
    x_fit=None,
    band="sd",
    ci_level=0.95,
    sd_multiplier=1.0,
    hac_maxlags=10,
):
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

    y_fit = np.asarray(
        result.predict(X_fit),
        dtype=float,
    )

    residuals = y - np.asarray(
        result.predict(X),
        dtype=float,
    )

    if result.df_resid > 0:
        residual_sd = np.sqrt(
            np.sum(residuals ** 2) / result.df_resid
        )
    else:
        residual_sd = np.nan

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
        ].to_numpy(dtype=float)

        band_upper = prediction_frame[
            "mean_ci_upper"
        ].to_numpy(dtype=float)

    else:
        band_lower = None
        band_upper = None

    return {
        "x_fit": x_fit,
        "y_fit": y_fit,
        "band_lower": band_lower,
        "band_upper": band_upper,
        "band_type": band,
        "intercept": result.params[0],
        "slope": result.params[1],
        "slope_pvalue": result.pvalues[1],
        "r_squared": result.rsquared,
        "residual_sd": residual_sd,
    }


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


def _collect_participant_emg_feature_regressions(
    features,
    y_col,
    x_col="trial_num",
    hac_maxlags=10,
    band="sd",
    sd_multiplier=1.0,
    ci_level=0.95,
):
    regressions = []

    participants = sorted(
        features["participant_id"].dropna().unique(),
        key=str,
    )

    for participant_id in participants:
        dfp = (
            features[
                features["participant_id"] == participant_id
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


def _plot_combined_emg_feature_regressions(
    regressions,
    plots_dir,
    subject_colors,
    filename,
    title,
    y_label,
    x_label="Trial number",
    xlim=(0.5, 400.5),
    block_size=40,
    alpha=0.05,
    band="sd",
    sd_multiplier=1.0,
    ci_level=0.95,
    show_zero_line=True,
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

    # Draw bands first
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

    # Draw regression lines
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
        for boundary in np.arange(block_size, xlim[1], block_size):
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
        title="Session-level HAC regressions",
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


def plot_emg_feature_regressions_combined(
    features,
    plots_dir,
    subject_colors,
    y_col,
    y_label,
    x_col="trial_num",
    x_label="Global trial number",
    xlim=(0.5, 400.5),
    block_size=40,
    fig_prefix="emg_feature_regressions_combined",
    alpha=0.05,
    hac_maxlags=10,
    band="sd",
    sd_multiplier=1.0,
    ci_level=0.95,
    split_by_isi=False,
):
    """
    Combined session-level regression plot for EMG features.

    If split_by_isi=False:
        one figure with all trials merged per session.

    If split_by_isi=True:
        one combined figure per ISI condition.
    """

    features = _prepare_emg_feature_x_column(
        features,
        x_col=x_col,
    )

    required_cols = {
        "participant_id",
        x_col,
        y_col,
    }

    if split_by_isi:
        required_cols.add("isi_bin")

    missing = required_cols.difference(features.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    if split_by_isi:
        isi_values = sorted(
            features["isi_bin"].dropna().unique(),
            key=_condition_sort_key,
        )

        for isi in isi_values:
            df_condition = features[
                features["isi_bin"] == isi
            ].copy()

            condition_label = _format_condition_label(isi)
            condition_safe = _make_safe_filename(condition_label)

            regressions = _collect_participant_emg_feature_regressions(
                features=df_condition,
                y_col=y_col,
                x_col=x_col,
                hac_maxlags=hac_maxlags,
                band=band,
                sd_multiplier=sd_multiplier,
                ci_level=ci_level,
            )

            _plot_combined_emg_feature_regressions(
                regressions=regressions,
                plots_dir=plots_dir,
                subject_colors=subject_colors,
                filename=f"{fig_prefix}_{condition_safe}.png",
                title=(
                    "Session regression lines for EMG feature — "
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
                show_zero_line=True,
                force_lower_zero=False,
            )

        return

    regressions = _collect_participant_emg_feature_regressions(
        features=features,
        y_col=y_col,
        x_col=x_col,
        hac_maxlags=hac_maxlags,
        band=band,
        sd_multiplier=sd_multiplier,
        ci_level=ci_level,
    )

    _plot_combined_emg_feature_regressions(
        regressions=regressions,
        plots_dir=plots_dir,
        subject_colors=subject_colors,
        filename=f"{fig_prefix}.png",
        title="Session regression lines for EMG feature",
        y_label=y_label,
        x_label=x_label,
        xlim=xlim,
        block_size=block_size,
        alpha=alpha,
        band=band,
        sd_multiplier=sd_multiplier,
        ci_level=ci_level,
        show_zero_line=True,
        force_lower_zero=False,
    )


def _get_robust_ylims(
    values,
    lower_pct=1,
    upper_pct=99,
    pad_fraction=0.20,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return -1, 1

    lower = np.nanpercentile(values, lower_pct)
    upper = np.nanpercentile(values, upper_pct)

    pad = pad_fraction * (upper - lower)

    if not np.isfinite(pad) or pad == 0:
        pad = 1.0

    return lower - pad, upper + pad


def _sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) <= 1:
        return np.nan

    return np.nanstd(values, ddof=1) / np.sqrt(len(values))


def plot_emg_feature_block_summary_combined(
    features,
    plots_dir,
    subject_colors,
    feature_col,
    y_label,
    fig_prefix="emg_feature_block_summary_combined",
):
    """
    Combined block-level plot.

    One figure.
    One line per session.
    """

    required_cols = {
        "participant_id",
        "trial_block",
        feature_col,
    }

    missing = required_cols.difference(features.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    session_ids = sorted(
        features["participant_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    if not session_ids:
        print(f"No sessions available for {feature_col}.")
        return

    fig, ax = plt.subplots(
        figsize=(11.5, 6.2),
    )

    all_y = []

    for session_id in session_ids:
        session_data = features[
            features["participant_id"] == session_id
        ].copy()

        summary = (
            session_data
            .groupby("trial_block", as_index=False)
            .agg(
                mean_value=(feature_col, "mean"),
                sem_value=(feature_col, _sem),
                n=(feature_col, lambda x: np.isfinite(x).sum()),
            )
        )

        summary = summary[
            np.isfinite(summary["mean_value"])
        ]

        if summary.empty:
            continue

        color = get_subject_color(
            session_id,
            subject_colors,
        )

        ax.errorbar(
            summary["trial_block"],
            summary["mean_value"],
            yerr=summary["sem_value"],
            color=color,
            marker="o",
            linewidth=2.6,
            markersize=6,
            capsize=3,
            label=session_id,
        )

        all_y.extend(
            summary["mean_value"].to_numpy(dtype=float)
        )

    ax.axhline(
        0,
        color="black",
        linestyle=":",
        linewidth=1.2,
        alpha=0.65,
        zorder=0,
    )

    if len(all_y) > 0:
        ax.set_ylim(
            *_get_robust_ylims(all_y)
        )

    ax.set_xlabel("Trial block")
    ax.set_ylabel(y_label)

    ax.set_title(
        "Block-level EMG feature summary\n"
        "Points = block mean | Error bars = SEM",
        fontweight="bold",
    )

    pretty_axes(ax)

    ax.legend(
        title="Session",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        right=0.76,
    )

    save_pretty_fig(
        fig,
        f"{fig_prefix}_{_make_safe_filename(feature_col)}.png",
        plots_dir,
    )