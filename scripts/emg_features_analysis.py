from pathlib import Path
import sys

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
FUNCTIONS_DIR = PROJECT_DIR / "functions"

if str(FUNCTIONS_DIR) not in sys.path:
    sys.path.append(str(FUNCTIONS_DIR))

from utils.config import get_config
from loading.load_data import load_timing_data, load_emg_accel_data
from preprocessing.isi_binning import add_isi_bin_column
from preprocessing.preprocess_emg import preprocess_emg_signal_table

from analysis.emg_features import compute_emg_trial_features

from plotting.plot_emg_features import (
    plot_emg_feature_regressions_combined,
    plot_emg_feature_block_summary_combined,

    # Optional individual plots, disabled by config for now
    plot_emg_feature_overlay,
    plot_emg_feature_block_summary,
)


C = get_config()

DATA_DIR = Path(C["data"]["path"]).expanduser()

if not DATA_DIR.exists():
    DATA_DIR = PROJECT_DIR / "data"

if not DATA_DIR.exists():
    raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

PLOTS_DIR = PROJECT_DIR / "outputs" / "plots" / "emg_features"
TABLES_DIR = PROJECT_DIR / "outputs" / "tables" / "emg_features"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def make_plot_column_and_label(
    feature_base,
    label_base,
    unit,
    normalization_method,
    n_baseline_trials,
):
    """
    Select which normalized column to plot.

    normalization_method options:
        "zscore"
        "percent"
        "centered"
        "raw"
    """

    if normalization_method == "zscore":
        return (
            f"{feature_base}_zscore_first{n_baseline_trials}",
            f"{label_base} z-score from first {n_baseline_trials} trials",
        )

    if normalization_method == "percent":
        return (
            f"{feature_base}_pct_first{n_baseline_trials}",
            f"{label_base} change from first {n_baseline_trials} trials (%)",
        )

    if normalization_method == "centered":
        return (
            f"{feature_base}_centered_first{n_baseline_trials}",
            f"{label_base} change from first {n_baseline_trials} trials ({unit})",
        )

    if normalization_method == "raw":
        return (
            feature_base,
            f"{label_base} ({unit})",
        )

    raise ValueError(
        f"Unknown normalization_method: {normalization_method}"
    )

def main():
    C = get_config()

    subject_colors = C["plot"]["subject_colors"]

    print("Loading timing data...")
    timing_data = load_timing_data(DATA_DIR)

    print("Loading EMG/accelerometer data...")
    signal_data = load_emg_accel_data(DATA_DIR)

    print("Adding ISI bins...")
    timing_data = add_isi_bin_column(timing_data, C)

    print("\nISI bins found:")
    print(
        timing_data
        .groupby("participant_id")["isi_bin"]
        .value_counts(dropna=False)
    )

    print("\nPreprocessing EMG...")
    processed_tables = []

    for participant_id, dfp in signal_data.groupby("participant_id", sort=False):
        print(f"Processing {participant_id}")

        dfp = dfp.sort_values("timestamp").copy()
        dfp = preprocess_emg_signal_table(dfp, C)

        processed_tables.append(dfp)

    signal_data_prep = pd.concat(
        processed_tables,
        ignore_index=True,
    )

    print("\nComputing trial-level EMG features...")
    features = compute_emg_trial_features(
        signal_data=signal_data_prep,
        timing_data=timing_data,
        config=C,
    )

    if features.empty:
        print("No EMG features were computed.")
        return

    output_file = TABLES_DIR / "emg_trial_features.csv"

    features.to_csv(
        output_file,
        index=False,
    )

    print(f"Saved EMG feature table to: {output_file}")

    print("\nFeature table overview:")
    print(features.groupby("participant_id").size())

    n_baseline_trials = C["emg_features"]["session_baseline"]["n_trials"]
    block_size = C["emg_features"]["plot"]["block_size"]
    band = C["emg_features"]["plot"]["band"]
    show_individual_points = C["emg_features"]["plot"]["show_individual_points"]

    # Main features to plot
    normalization_method = (
        C["emg_features"]
        .get("normalization", {})
        .get("plot_method", "zscore")
    )

    rolling_window_trials = (
        C["emg_features"]
        .get("variability", {})
        .get("rolling_window_trials", 10)
    )

    feature_specs = [
        {
            "feature_base": "rms_response",
            "label_base": "RMS",
            "unit": "µV",
            "fig_prefix": "rms_response",
        },
        {
            "feature_base": "rms_response_minus_pre_event",
            "label_base": "Baseline-corrected RMS",
            "unit": "µV",
            "fig_prefix": "rms_minus_pre_event",
        },
        {
            "feature_base": "iemg_response",
            "label_base": "iEMG",
            "unit": "µV·s",
            "fig_prefix": "iemg_response",
        },
        {
            "feature_base": "peak_abs_response",
            "label_base": "Peak EMG",
            "unit": "µV",
            "fig_prefix": "peak_abs_response",
        },
        {
            "feature_base": "mdf_response_hz",
            "label_base": "MDF",
            "unit": "Hz",
            "fig_prefix": "mdf_response",
        },
        {
            "feature_base": "mnf_response_hz",
            "label_base": "MNF",
            "unit": "Hz",
            "fig_prefix": "mnf_response",
        },

        # --------------------------------------------------------
        # Variability features
        # --------------------------------------------------------
        {
            "feature_base": f"rms_response_rolling_sd_{rolling_window_trials}",
            "label_base": f"RMS rolling SD ({rolling_window_trials} trials)",
            "unit": "µV",
            "fig_prefix": f"rms_response_rolling_sd_{rolling_window_trials}",
        },
        {
            "feature_base": f"rms_response_minus_pre_event_rolling_sd_{rolling_window_trials}",
            "label_base": f"Baseline-corrected RMS rolling SD ({rolling_window_trials} trials)",
            "unit": "µV",
            "fig_prefix": f"rms_minus_pre_event_rolling_sd_{rolling_window_trials}",
        },
    ]

    feature_plots = []

    for spec in feature_specs:
        feature_col, y_label = make_plot_column_and_label(
            feature_base=spec["feature_base"],
            label_base=spec["label_base"],
            unit=spec["unit"],
            normalization_method=normalization_method,
            n_baseline_trials=n_baseline_trials,
        )

        feature_plots.append(
            {
                "feature_col": feature_col,
                "y_label": y_label,
                "fig_prefix": (
                    spec["fig_prefix"]
                    + f"_{normalization_method}"
                    + f"_first{n_baseline_trials}"
                ),
            }
        )

    print("\nPlotting EMG feature trends...")

    plot_cfg = C["emg_features"]["plot"]

    n_baseline_trials = C["emg_features"]["session_baseline"]["n_trials"]

    block_size = plot_cfg.get("block_size", 40)
    band = plot_cfg.get("band", "sd")
    sd_multiplier = plot_cfg.get("sd_multiplier", 1.0)
    ci_level = plot_cfg.get("ci_level", 0.95)
    alpha = plot_cfg.get("alpha", 0.05)

    make_combined_regression_plots = plot_cfg.get(
        "make_combined_regression_plots",
        True,
    )

    make_combined_block_plots = plot_cfg.get(
        "make_combined_block_plots",
        True,
    )

    make_individual_plots = plot_cfg.get(
        "make_individual_plots",
        False,
    )

    make_split_by_isi_plots = plot_cfg.get(
        "make_split_by_isi_plots",
        True,
    )

    show_individual_points = plot_cfg.get(
        "show_individual_points",
        True,
    )

    for item in feature_plots:
        feature_col = item["feature_col"]

        if feature_col not in features.columns:
            print(f"Skipping missing feature: {feature_col}")
            continue

        print(f"Plotting {feature_col}")

        # ------------------------------------------------------------
        # Main combined regression plot:
        # all sessions on the same plot
        # ------------------------------------------------------------
        if make_combined_regression_plots:
            plot_emg_feature_regressions_combined(
                features=features,
                plots_dir=PLOTS_DIR,
                subject_colors=subject_colors,
                y_col=feature_col,
                y_label=item["y_label"],
                x_col="trial_num",
                x_label="Global trial number",
                xlim=(0.5, 400.5),
                block_size=block_size,
                fig_prefix=item["fig_prefix"] + "_combined_regression",
                alpha=alpha,
                hac_maxlags=10,
                band=band,
                sd_multiplier=sd_multiplier,
                ci_level=ci_level,
                split_by_isi=False,
            )

        # ------------------------------------------------------------
        # Optional combined regression split by ISI:
        # one combined figure per ISI
        # ------------------------------------------------------------
        if make_split_by_isi_plots:
            plot_emg_feature_regressions_combined(
                features=features,
                plots_dir=PLOTS_DIR,
                subject_colors=subject_colors,
                y_col=feature_col,
                y_label=item["y_label"],
                x_col="trial_num",
                x_label="Global trial number",
                xlim=(0.5, 400.5),
                block_size=block_size,
                fig_prefix=item["fig_prefix"] + "_combined_regression_by_isi",
                alpha=alpha,
                hac_maxlags=10,
                band=band,
                sd_multiplier=sd_multiplier,
                ci_level=ci_level,
                split_by_isi=True,
            )

        # ------------------------------------------------------------
        # Combined block-level summary:
        # all sessions on one plot
        # ------------------------------------------------------------
        if make_combined_block_plots:
            plot_emg_feature_block_summary_combined(
                features=features,
                plots_dir=PLOTS_DIR,
                subject_colors=subject_colors,
                feature_col=feature_col,
                y_label=item["y_label"],
                fig_prefix=item["fig_prefix"] + "_combined_block_summary",
            )

        # ------------------------------------------------------------
        # Optional individual plots.
        # Disabled by default in config.
        # ------------------------------------------------------------
        if make_individual_plots:
            plot_emg_feature_overlay(
                features=features,
                plots_dir=PLOTS_DIR,
                subject_colors=subject_colors,
                feature_col=feature_col,
                y_label=item["y_label"],
                fig_prefix=item["fig_prefix"] + "_individual_overlay_all_isi",
                block_size=block_size,
                xlim=(0.5, 400.5),
                band=band,
                show_individual_points=show_individual_points,
                facet_by_isi=False,
            )

            plot_emg_feature_block_summary(
                features=features,
                plots_dir=PLOTS_DIR,
                subject_colors=subject_colors,
                feature_col=feature_col,
                y_label=item["y_label"],
                fig_prefix=item["fig_prefix"] + "_individual_block_summary",
            )


if __name__ == "__main__":
    main()