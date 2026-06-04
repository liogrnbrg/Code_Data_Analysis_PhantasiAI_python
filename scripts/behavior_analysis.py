from pathlib import Path
import sys
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
FUNCTIONS_DIR = PROJECT_DIR / "functions"

sys.path.append(str(FUNCTIONS_DIR))

from utils.config import get_config
from loading.load_data import load_timing_data, load_emg_accel_data
from preprocessing.isi_binning import add_isi_bin_column
from preprocessing.accel_features import (
    add_accel_norm,
    extract_baseline_corrected_movement_metrics,
    extract_integrated_accel_displacement_metrics,
)
from plotting.plot_behavior_features import (
    plot_behavior_metric_over_trials,
    plot_behavior_metric_by_isi,
    plot_emg_vs_behavior_metric,
)


def main():

    C = get_config()

    DATA_DIR = Path(C["data"]["path"]).expanduser()
    DATA_DIR = DATA_DIR if DATA_DIR.exists() else PROJECT_DIR / "data"

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    PLOTS_DIR = PROJECT_DIR / "outputs" / "plots" / "behavior_analysis"
    TABLES_DIR = PROJECT_DIR / "outputs" / "tables"

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    subject_colors = C["plot"]["subject_colors"]

    # ---------- Load data ----------
    print("Loading data...")
    timing_data = load_timing_data(DATA_DIR)
    accel_data = load_emg_accel_data(DATA_DIR)

    timing_data = add_isi_bin_column(timing_data, C)

    print("Preparing acceleration features...")
    accel_data = add_accel_norm(accel_data)

    # ---------- Movement metrics ----------
    print("Extracting baseline-corrected movement metrics...")
    movement_metrics = extract_baseline_corrected_movement_metrics(
        accel_data=accel_data,
        timing_data=timing_data,
        baseline_window_s=(-0.5, 0),
        response_window_s=(0, None),
    )

    movement_metrics.to_csv(TABLES_DIR / "movement_metrics.csv", index=False)
    print("Saved:", TABLES_DIR / "movement_metrics.csv")

    # ---------- Double integration displacement proxy ----------
    print("Extracting exploratory double-integration displacement metrics...")

    all_disp_metrics = []

    for axis in ["accel_x", "accel_y", "accel_z"]:
        disp_axis = extract_integrated_accel_displacement_metrics(
            accel_data=accel_data,
            timing_data=timing_data,
            accel_axis=axis,
            baseline_window_s=(-0.5, 0),
            response_window_s=(0, None),
            detrend_position=True,
        )

        if disp_axis.empty:
            continue

        disp_axis = disp_axis.rename(columns={
            "disp_range_proxy": f"{axis}_disp_range_proxy",
            "disp_peak_abs_proxy": f"{axis}_disp_peak_abs_proxy",
            "disp_final_proxy": f"{axis}_disp_final_proxy",
            "disp_peak_latency_s": f"{axis}_disp_peak_latency_s",
        })

        keep_cols = [
            "participant_id",
            "trial_num",
            "isi",
            "isi_bin",
            "event",
            f"{axis}_disp_range_proxy",
            f"{axis}_disp_peak_abs_proxy",
            f"{axis}_disp_final_proxy",
            f"{axis}_disp_peak_latency_s",
        ]

        all_disp_metrics.append(disp_axis[keep_cols])

    if len(all_disp_metrics) > 0:
        disp_metrics = all_disp_metrics[0]

        for df in all_disp_metrics[1:]:
            disp_metrics = disp_metrics.merge(
                df,
                on=["participant_id", "trial_num", "isi", "isi_bin", "event"],
                how="outer",
            )
    else:
        disp_metrics = pd.DataFrame()

    disp_metrics.to_csv(TABLES_DIR / "displacement_proxy_metrics.csv", index=False)
    print("Saved:", TABLES_DIR / "displacement_proxy_metrics.csv")

    # ---------- EMG metrics ----------
    print("Preparing EMG timing metrics...")

    emg_metrics = timing_data[
        ["participant_id", "trial_num", "peak_amp", "peak_time", "event"]
    ].copy()

    emg_metrics["emg_peak_delay_s"] = (
        emg_metrics["peak_time"] - emg_metrics["event"]
    )

    # ---------- Merge behavior + EMG ----------
    print("Merging movement metrics with EMG metrics...")

    movement_emg = movement_metrics.merge(
        emg_metrics,
        on=["participant_id", "trial_num", "event"],
        how="left",
    )

    movement_emg.to_csv(TABLES_DIR / "movement_emg_metrics.csv", index=False)
    print("Saved:", TABLES_DIR / "movement_emg_metrics.csv")

    if not disp_metrics.empty:
        disp_emg = disp_metrics.merge(
            emg_metrics,
            on=["participant_id", "trial_num", "event"],
            how="left",
        )

        disp_emg.to_csv(TABLES_DIR / "displacement_proxy_emg_metrics.csv", index=False)
        print("Saved:", TABLES_DIR / "displacement_proxy_emg_metrics.csv")
    else:
        disp_emg = pd.DataFrame()

    # ---------- Plot movement metrics ----------
    movement_plot_metrics = [
        "movement_peak",
        "movement_peak_latency_s",
        "movement_auc",
        "movement_mean",
        "movement_sd",
    ]

    for metric in movement_plot_metrics:
        if metric not in movement_metrics.columns:
            continue

        print(f"Plotting {metric}...")

        plot_behavior_metric_over_trials(
            behavior_data=movement_metrics,
            metric=metric,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
        )

        plot_behavior_metric_by_isi(
            behavior_data=movement_metrics,
            metric=metric,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
        )

    # ---------- Plot displacement proxy metrics ----------
    disp_plot_metrics = [
        "accel_x_disp_range_proxy",
        "accel_y_disp_range_proxy",
        "accel_z_disp_range_proxy",
        "accel_x_disp_peak_abs_proxy",
        "accel_y_disp_peak_abs_proxy",
        "accel_z_disp_peak_abs_proxy",
    ]

    for metric in disp_plot_metrics:
        if metric not in disp_metrics.columns:
            continue

        print(f"Plotting {metric}...")

        plot_behavior_metric_over_trials(
            behavior_data=disp_metrics,
            metric=metric,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
        )

        plot_behavior_metric_by_isi(
            behavior_data=disp_metrics,
            metric=metric,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
        )

    # ---------- Plot EMG vs movement ----------
    movement_relation_pairs = [
        ("peak_amp", "movement_peak"),
        ("peak_amp", "movement_auc"),
        ("peak_amp", "movement_mean"),
        ("emg_peak_delay_s", "movement_peak_latency_s"),
    ]

    for emg_metric, behavior_metric in movement_relation_pairs:
        if emg_metric not in movement_emg.columns or behavior_metric not in movement_emg.columns:
            continue

        print(f"Plotting {emg_metric} vs {behavior_metric}...")

        plot_emg_vs_behavior_metric(
            merged_data=movement_emg,
            emg_metric=emg_metric,
            behavior_metric=behavior_metric,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
        )

    # ---------- Plot EMG vs displacement proxy ----------
    if not disp_emg.empty:
        disp_relation_pairs = [
            ("peak_amp", "accel_x_disp_range_proxy"),
            ("peak_amp", "accel_y_disp_range_proxy"),
            ("peak_amp", "accel_z_disp_range_proxy"),
            ("emg_peak_delay_s", "accel_x_disp_peak_latency_s"),
            ("emg_peak_delay_s", "accel_y_disp_peak_latency_s"),
            ("emg_peak_delay_s", "accel_z_disp_peak_latency_s"),
        ]

        for emg_metric, behavior_metric in disp_relation_pairs:
            if emg_metric not in disp_emg.columns or behavior_metric not in disp_emg.columns:
                continue

            print(f"Plotting {emg_metric} vs {behavior_metric}...")

            plot_emg_vs_behavior_metric(
                merged_data=disp_emg,
                emg_metric=emg_metric,
                behavior_metric=behavior_metric,
                plots_dir=PLOTS_DIR,
                subject_colors=subject_colors,
            )

    print("Done.")


if __name__ == "__main__":
    main()