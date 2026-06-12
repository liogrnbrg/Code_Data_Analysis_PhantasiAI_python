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
    add_trialwise_velocity_position_proxies,
    extract_trial_amplitude_metrics,
)
from plotting.plot_behavior_features import (
    plot_behavior_metric_over_trials,
    plot_behavior_metric_by_isi,
    plot_emg_vs_behavior_metric,
)
from plotting.animate_trajectories import (
    animate_trial_trajectory_2d, 
    animate_trial_trajectory_3d,
    animate_trial_sequence_2d
)

from preprocessing.preprocess_accel import preprocess_accel_signal_table


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
    signal_data = load_emg_accel_data(DATA_DIR)

    print(f"Found participants in timing data: {timing_data['participant_id'].unique()}")
    print(f"Found participants in signal data: {signal_data['participant_id'].unique()}")
    
    timing_data = add_isi_bin_column(timing_data, C)

    accel_preprocess_enabled = C.get("accel_preprocess", {}).get("enabled", False)

    if accel_preprocess_enabled:
        print("Preprocessing acceleration...")
        signal_data = preprocess_accel_signal_table(signal_data, C)

        accel_cols_for_analysis = (
            "accel_x_preprocessed",
            "accel_y_preprocessed",
            "accel_z_preprocessed",
        )
    else:
        print("Skipping acceleration preprocessing. Using raw acceleration.")

        accel_cols_for_analysis = (
            "accel_x",
            "accel_y",
            "accel_z",
        )

    print("Acceleration columns used for integration and behavior metrics:")
    print(accel_cols_for_analysis)

    print("Computing trial-wise velocity and position proxies...")
    signal_data = add_trialwise_velocity_position_proxies(
        data=signal_data,
        timing_data=timing_data,
        accel_cols=accel_cols_for_analysis,
        axis_names=("x", "y", "z"),
        baseline_window_s=(-0.5, -0.1),
        force_zero_velocity_end=True,
        detrend_position=True,
    )

    print("Extracting per-trial amplitude metrics...")
    behavior_metrics = extract_trial_amplitude_metrics(
        signal_data=signal_data,
        timing_data=timing_data,
        signal_var_map={
            "accel_x": accel_cols_for_analysis[0],
            "accel_y": accel_cols_for_analysis[1],
            "accel_z": accel_cols_for_analysis[2],

            "velocity_x": "velocity_x",
            "velocity_y": "velocity_y",
            "velocity_z": "velocity_z",

            "position_x": "position_x",
            "position_y": "position_y",
            "position_z": "position_z",
        },
    )

    behavior_metrics.to_csv(TABLES_DIR / "behavior_amplitude_metrics.csv", index=False)
    print("Saved:", TABLES_DIR / "behavior_amplitude_metrics.csv")

    print("Preparing EMG metrics...")
    emg_metrics = timing_data[
        ["participant_id", "trial_num", "peak_amp", "peak_time", "event"]
    ].copy()

    emg_metrics["emg_peak_delay_s"] = (
        emg_metrics["peak_time"] - emg_metrics["event"]
    )

    merged = behavior_metrics.merge(
        emg_metrics,
        on=["participant_id", "trial_num", "event"],
        how="left",
    )

    merged.to_csv(TABLES_DIR / "behavior_emg_amplitude_metrics.csv", index=False)
    print("Saved:", TABLES_DIR / "behavior_emg_amplitude_metrics.csv")

    # ---------- Main behavioral amplitude plots ----------
    behavior_plot_metrics = [
        "accel_x_amp",
        "accel_y_amp",
        "accel_z_amp",
        "accel_3d_amp",
        "accel_session_dominant_amp",

        "velocity_x_amp",
        "velocity_y_amp",
        "velocity_z_amp",
        "velocity_3d_amp",
        "velocity_session_dominant_amp",

        "position_x_amp",
        "position_y_amp",
        "position_z_amp",
        "position_3d_amp",
        "position_session_dominant_amp",
    ]

    for metric in behavior_plot_metrics:

        if metric not in behavior_metrics.columns:
            print(f"Skipping {metric}, not found.")
            continue

        print(f"Plotting {metric} over trials...")
        plot_behavior_metric_over_trials(
            behavior_data=behavior_metrics,
            metric=metric,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
        )

        print(f"Plotting {metric} by ISI...")
        plot_behavior_metric_by_isi(
            behavior_data=behavior_metrics,
            metric=metric,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
        )

    # ---------- EMG vs behavioral amplitude ----------
    relation_pairs = [
        ("peak_amp", "accel_3d_amp"),
        ("peak_amp", "velocity_3d_amp"),
        ("peak_amp", "position_3d_amp"),

        ("peak_amp", "accel_session_dominant_amp"),
        ("peak_amp", "velocity_session_dominant_amp"),
        ("peak_amp", "position_session_dominant_amp"),

        ("emg_peak_delay_s", "velocity_3d_amp"),
        ("emg_peak_delay_s", "position_3d_amp"),
    ]

    for emg_metric, behavior_metric in relation_pairs:

        if emg_metric not in merged.columns or behavior_metric not in merged.columns:
            print(f"Skipping {emg_metric} vs {behavior_metric}, missing column.")
            continue

        print(f"Plotting {emg_metric} vs {behavior_metric}...")
        plot_emg_vs_behavior_metric(
            merged_data=merged,
            emg_metric=emg_metric,
            behavior_metric=behavior_metric,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
        )

    print("Plots saved to:", PLOTS_DIR)


    GIF_DIR = PROJECT_DIR / "outputs" / "gifs" / "behavior_trajectories"
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    participant_axes = {
        "Lio": ("position_y", "position_z"),
        "Parisa": ("position_x", "position_z"),
    }

    # ---------- Individual trial GIFs ----------
    for participant_id, axes in participant_axes.items():

        for trial_num in [10, 100, 200, 300]:

            animate_trial_trajectory_2d(
                signal_data=signal_data,
                timing_data=timing_data,
                participant_id=participant_id,
                trial_num=trial_num,
                plots_dir=GIF_DIR,
                subject_colors=subject_colors,
                x_var=axes[0],
                y_var=axes[1],
                fps=20,
                n_frames=120,
            )

            animate_trial_trajectory_3d(
                signal_data=signal_data,
                timing_data=timing_data,
                participant_id=participant_id,
                trial_num=trial_num,
                plots_dir=GIF_DIR,
                subject_colors=subject_colors,
                x_var="position_x",
                y_var="position_y",
                z_var="position_z",
                fps=20,
                n_frames=120,
                elev=25,
                azim=45,
            )

    # ---------- Sequence GIFs: 10 trials in one GIF ----------
    for participant_id, axes in participant_axes.items():

        animate_trial_sequence_2d(
            signal_data=signal_data,
            timing_data=timing_data,
            participant_id=participant_id,
            trial_nums=list(range(1, 11)),
            plots_dir=GIF_DIR,
            subject_colors=subject_colors,
            x_var=axes[0],
            y_var=axes[1],
            fps=20,
            frames_per_trial=50,
            pause_frames=8,
            reset_to_origin=True,
        )

    print("GIFs saved to:", GIF_DIR)

if __name__ == "__main__":
    main()

