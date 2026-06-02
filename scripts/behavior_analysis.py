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
    add_accel_tilt_angles,
    extract_accel_behavior_metrics,
    extract_angle_rom_metrics,
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

    print("Loading data...")
    timing_data = load_timing_data(DATA_DIR)
    accel_data = load_emg_accel_data(DATA_DIR)

    timing_data = add_isi_bin_column(timing_data, C)

    print("Preparing acceleration features...")
    accel_data = add_accel_norm(accel_data)
    accel_data = add_accel_tilt_angles(accel_data)

    print("Extracting acceleration behavioral metrics...")
    accel_metrics = extract_accel_behavior_metrics(
        accel_data=accel_data,
        timing_data=timing_data,
        signal_var="accel_norm",
        baseline_window_s=(-0.5, 0),
        response_window_s=(0, None),
    )

    print("Extracting angle / ROM proxy metrics...")
    rom_metrics = extract_angle_rom_metrics(
        accel_data=accel_data,
        timing_data=timing_data,
        angle_vars=("tilt_x_deg", "tilt_y_deg", "tilt_z_deg"),
    )

    behavior_metrics = accel_metrics.merge(
        rom_metrics,
        on=["participant_id", "trial_num", "isi", "isi_bin", "event"],
        how="outer",
    )

    behavior_metrics.to_csv(TABLES_DIR / "behavior_metrics.csv", index=False)

    print("Behavior metrics saved to:")
    print(TABLES_DIR / "behavior_metrics.csv")

    print("Merging with EMG timing metrics...")
    emg_cols = [
        "participant_id",
        "trial_num",
        "peak_amp",
        "peak_time",
        "event",
    ]

    emg_metrics = timing_data[emg_cols].copy()
    emg_metrics["emg_peak_delay_s"] = emg_metrics["peak_time"] - emg_metrics["event"]

    merged = behavior_metrics.merge(
        emg_metrics,
        on=["participant_id", "trial_num", "event"],
        how="left",
    )

    merged.to_csv(TABLES_DIR / "behavior_emg_metrics.csv", index=False)

    print("Merged behavior/EMG metrics saved to:")
    print(TABLES_DIR / "behavior_emg_metrics.csv")

    # ---------- Behavior plots ----------
    behavior_plot_metrics = [
        "accel_peak_relative",
        "accel_peak_latency_s",
        "accel_auc_relative",
        "accel_range",
        "tilt_x_deg_rom_proxy",
        "tilt_y_deg_rom_proxy",
        "tilt_z_deg_rom_proxy",
    ]

    for metric in behavior_plot_metrics:
        if metric not in behavior_metrics.columns:
            continue

        print(f"Plotting {metric}...")
        plot_behavior_metric_over_trials(
            behavior_data=behavior_metrics,
            metric=metric,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
        )

        plot_behavior_metric_by_isi(
            behavior_data=behavior_metrics,
            metric=metric,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
        )

    # ---------- EMG vs behavior plots ----------
    relation_pairs = [
        ("peak_amp", "accel_peak_relative"),
        ("peak_amp", "accel_range"),
        ("emg_peak_delay_s", "accel_peak_latency_s"),
        ("peak_amp", "tilt_x_deg_rom_proxy"),
        ("peak_amp", "tilt_y_deg_rom_proxy"),
        ("peak_amp", "tilt_z_deg_rom_proxy"),
    ]

    for emg_metric, behavior_metric in relation_pairs:
        if emg_metric not in merged.columns or behavior_metric not in merged.columns:
            continue

        print(f"Plotting {emg_metric} vs {behavior_metric}...")
        plot_emg_vs_behavior_metric(
            merged_data=merged,
            emg_metric=emg_metric,
            behavior_metric=behavior_metric,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
        )

    print("Done.")


if __name__ == "__main__":
    main()