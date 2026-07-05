# stim_vs_nostim_comparison.py

from pathlib import Path
import sys

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
FUNCTIONS_DIR = PROJECT_DIR / "functions"

if str(FUNCTIONS_DIR) not in sys.path:
    sys.path.append(str(FUNCTIONS_DIR))


from loading.load_data import load_data
from preprocessing.isi_binning import add_isi_bin_column
from preprocessing.preprocess_emg import preprocess_emg_signal_table
from preprocessing.reaction_time import (
    extract_emg_reaction_times,
)

from analysis.stim_nostim_comparison import (
    prepare_rt_stim_nostim_comparison,
    add_rolling_rt_variability,
    prepare_variability_pairs,
    compute_rt_paired_statistics,
    compute_rt_block_summary,
    compute_paired_block_statistics,
    compute_stim_nostim_trend_statistics,
    compute_stim_nostim_variability_trend_statistics,
)

from plotting.plot_stim_nostim_comparison import (
    plot_stim_nostim_rt_overlay,
    plot_stim_minus_nostim_rt_difference,
    plot_stim_nostim_rt_variability,
)

from utils.config import get_config


def main():

    print("Loading configuration...")
    C = get_config()
    subject_colors = C["plot"]["subject_colors"]

    output_dir = PROJECT_DIR / "outputs"

    tables_dir = output_dir / "tables" / "stim_vs_nostim"
    plots_dir = output_dir / "plots" / "stim_vs_nostim"

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------
    print("Loading data...")

    data_dir = Path(C["data"]["path"]).expanduser()

    # Adjust this line to match the working call used in
    # reaction_time_analysis.py.
    signal_data, timing_data = load_data(data_dir)

    timing_data = add_isi_bin_column(
        timing_data,
        C,
    )

    # -------------------------------------------------------------
    # 2. Preprocess EMG
    # -------------------------------------------------------------
    print("Preprocessing EMG...")

    processed_sessions = []

    for participant_id, signal_participant in signal_data.groupby(
        "participant_id",
        sort=False,
    ):
        print(f"Processing {participant_id}")

        signal_participant_processed = preprocess_emg_signal_table(
            df=signal_participant,
            config=C,
        )

        processed_sessions.append(signal_participant_processed)

    signal_data = pd.concat(
        processed_sessions,
        ignore_index=True,
    )

    emg_var = C["emg_patterns"]["preprocess"]["output_var"]
    # -------------------------------------------------------------
    # 3. Extract reaction times
    # -------------------------------------------------------------
    print("Extracting reaction times...")

    rt_data = extract_emg_reaction_times(
        signal_data=signal_data,
        timing_data=timing_data,
        emg_var=emg_var,
        baseline_window_s=(-0.5, -0.1),
        response_window_s=(0.0, 1.5),
        threshold_sd=2.0,
        smooth_window_s=0.05,
        onset_fraction=0.20,
        min_peak_prominence_sd=3.0,
        rectify=True,
    )

    rt_data.to_csv(
        tables_dir / "reaction_time_all_sessions.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # 4. Prepare STIM/NOSTIM paired data
    # -------------------------------------------------------------
    print("Preparing paired STIM/NOSTIM comparison...")

    long_data, paired_data = prepare_rt_stim_nostim_comparison(
        rt_data=rt_data,
        rt_col="reaction_time_ms",
        n_baseline_trials=10,
        config=C,
    )

    print("\nSessions detected:")
    print(
        long_data[
            [
                "participant_id",
                "base_participant",
                "condition",
                "condition_label",
                "session_number",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "base_participant",
                "condition",
                "session_number",
            ]
        )
        .to_string(index=False)
    )

    print("\nComparisons created:")
    if paired_data.empty:
        print("No paired comparisons available.")
    else:
        print(
            paired_data[
                [
                    "comparison_label",
                    "base_participant",
                    "test_session_id",
                    "reference_session_id",
                    "isi_bin",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "base_participant",
                    "comparison_label",
                    "isi_bin",
                ]
            )
            .to_string(index=False)
        )

    long_data = add_rolling_rt_variability(
        long_data=long_data,
        rt_col="reaction_time_ms",
        rolling_window=5,
        min_periods=4,
    )

    variability_pairs = prepare_variability_pairs(
        long_data=long_data,
        config=C,
    )

    long_data.to_csv(
        tables_dir / "stim_nostim_rt_long.csv",
        index=False,
    )

    paired_data.to_csv(
        tables_dir / "stim_nostim_rt_paired.csv",
        index=False,
    )

    variability_pairs.to_csv(
        tables_dir / "stim_nostim_rt_variability_paired.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # 5. Statistics
    # -------------------------------------------------------------

    print("Comparing STIM and NOSTIM variability trends...")

    variability_trend_stats = (
        compute_stim_nostim_variability_trend_statistics(
            long_data=long_data,
            variability_col="reaction_time_rolling_sd_ms",
            hac_maxlags=10,
            config=C,
        )
    )

    variability_trend_stats.to_csv(
        tables_dir / "stim_nostim_rt_variability_trend_comparison.csv",
        index=False,
    )

    print("\nCondition variability trend comparison:")
    print(
        variability_trend_stats[
            [
                "comparison_label",
                "base_participant",
                "isi_bin",
                "n_test",
                "n_reference",
                "test_slope_sd_ms_per_trial",
                "reference_slope_sd_ms_per_trial",
                "interaction_slope_difference_sd_ms_per_trial",
                "interaction_pvalue",
            ]
        ].to_string(index=False)
    )
    print("Comparing STIM and NOSTIM temporal trends...")

    trend_stats = compute_stim_nostim_trend_statistics(
        long_data=long_data,
        y_col="reaction_time_centered_ms",
        hac_maxlags=10,
        config=C,
    )

    trend_stats.to_csv(
        tables_dir / "stim_nostim_rt_trend_comparison.csv",
        index=False,
    )

    print("\nCondition trend comparison:")
    print(
        trend_stats[
            [
                "comparison_label",
                "base_participant",
                "isi_bin",
                "n_test",
                "n_reference",
                "test_slope_ms_per_trial",
                "reference_slope_ms_per_trial",
                "interaction_slope_difference_ms_per_trial",
                "interaction_pvalue",
            ]
        ].to_string(index=False)
    )

    print("Computing paired statistics...")

    overall_stats = compute_rt_paired_statistics(
        paired_data=paired_data,
        difference_col="rt_difference_centered_ms",
    )

    block_summary = compute_rt_block_summary(
        long_data=long_data,
        rt_col="reaction_time_ms",
    )

    block_stats = compute_paired_block_statistics(
        paired_data=paired_data,
        difference_col="rt_difference_centered_ms",
    )

    overall_stats.to_csv(
        tables_dir / "stim_nostim_rt_statistics_by_isi.csv",
        index=False,
    )

    block_summary.to_csv(
        tables_dir / "stim_nostim_rt_block_summary.csv",
        index=False,
    )

    block_stats.to_csv(
        tables_dir / "stim_nostim_rt_block_statistics.csv",
        index=False,
    )

    print("\nPaired RT statistics by comparison and ISI:")
    print(
        overall_stats[
            [
                "comparison_label",
                "base_participant",
                "isi_bin",
                "n_valid_pairs",
                "mean_difference_test_minus_reference_ms",
                "cohens_dz",
                "paired_t_pvalue",
                "wilcoxon_pvalue",
                "difference_slope_ms_per_trial",
                "difference_slope_pvalue",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------
    # 6. Figures
    # -------------------------------------------------------------
    print("Plotting STIM versus NOSTIM RT trajectories...")

    plot_stim_nostim_rt_overlay(
        long_data=long_data,
        trend_stats=trend_stats,
        plots_dir=plots_dir,
        subject_colors=subject_colors,
        y_col="reaction_time_centered_ms",
        y_label="RT change from first 10 trials (ms)",
        block_size=80,
        xlim=(0.5, 400.5),
        fig_prefix="stim_vs_nostim_centered_rt",
        config=C,
    )

    plot_stim_nostim_rt_overlay(
        long_data=long_data,
        trend_stats=trend_stats,
        plots_dir=plots_dir,
        subject_colors=subject_colors,
        y_col="reaction_time_centered_ms",
        y_label="RT change from first 10 trials (ms)",
        block_size=80,
        xlim=(0.5, 400.5),
        fig_prefix="stim_vs_nostim_centered_rt_no_points",
        show_individual_points=False,
        config=C,
    )

    print("Plotting paired STIM minus NOSTIM differences...")

    plot_stim_minus_nostim_rt_difference(
        paired_data=paired_data,
        plots_dir=plots_dir,
        value_col="rt_difference_centered_ms",
        y_label="Test − reference centered RT (ms)",
        block_size=80,
        xlim=(0.5, 400.5),
        fig_prefix="condition_difference_centered_rt",
        config=C,
    )

    print("Plotting RT variability comparison...")

    plot_stim_nostim_rt_variability(
        long_data=long_data,
        variability_trend_stats=variability_trend_stats,
        plots_dir=plots_dir,
        subject_colors=subject_colors,
        variability_col="reaction_time_rolling_sd_ms",
        block_size=80,
        xlim=(0.5, 400.5),
        fig_prefix="stim_vs_nostim_rt_variability",
        config=C,
    )

    plot_stim_nostim_rt_variability(
        long_data=long_data,
        variability_trend_stats=variability_trend_stats,
        plots_dir=plots_dir,
        subject_colors=subject_colors,
        variability_col="reaction_time_rolling_sd_ms",
        block_size=80,         
        xlim=(0.5, 400.5),
        fig_prefix="stim_vs_nostim_rt_variability_no_points",
        show_individual_points=False,
        config=C,
    )

    print("\nDone.")
    print(f"Tables saved to:\n{tables_dir}")
    print(f"Plots saved to:\n{plots_dir}")


if __name__ == "__main__":
    main()