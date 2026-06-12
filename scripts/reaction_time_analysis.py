from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
FUNCTIONS_DIR = PROJECT_DIR / "functions"

sys.path.append(str(FUNCTIONS_DIR))

from utils.config import get_config
from loading.load_data import load_timing_data, load_emg_accel_data
from preprocessing.isi_binning import add_isi_bin_column
from preprocessing.preprocess_emg import preprocess_emg_signal_table
from preprocessing.reaction_time import (
    extract_emg_reaction_times,
    add_reaction_time_normalization,
)
from plotting.plot_reaction_time import (
    plot_reaction_time_one_figure_per_participant,
    plot_reaction_time_variability_one_figure_per_participant,
    quick_plot_rt_detection,
)


def main():

    C = get_config()

    DATA_DIR = Path(C["data"]["path"]).expanduser()
    print(f"Using DATA_DIR: {DATA_DIR}")
    DATA_DIR = DATA_DIR if DATA_DIR.exists() else PROJECT_DIR / "data"

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    PLOTS_DIR = PROJECT_DIR / "outputs" / "plots" / "reaction_time"
    TABLES_DIR = PROJECT_DIR / "outputs" / "tables"

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    subject_colors = C["plot"]["subject_colors"]

    print("Loading data...")
    timing_data = load_timing_data(DATA_DIR)
    signal_data = load_emg_accel_data(DATA_DIR)

    timing_data = add_isi_bin_column(timing_data, C)

    # For STIM sessions, the stimulation condition is stored in the "stim" column.
    # Use it as isi_bin so the reaction-time plots are grouped by stimulation condition.
    if "stim" in timing_data.columns:
        stim_available = timing_data["stim"].notna()

        print("Using stim column as isi_bin for rows with available stim values:")
        print(timing_data.loc[stim_available, "participant_id"].value_counts())

        timing_data.loc[stim_available, "isi_bin"] = timing_data.loc[stim_available, "stim"]
    
    print("ISI / ISI_bin counts by participant:")
    print(
        timing_data
        .groupby("participant_id")[["isi", "isi_bin"]]
        .agg(
            n_trials=("isi", "size"),
            n_isi_non_nan=("isi", lambda x: x.notna().sum()),
            n_isi_bin_non_nan=("isi_bin", lambda x: x.notna().sum()),
            unique_isi=("isi", lambda x: sorted(x.dropna().unique())[:20]),
            unique_isi_bin=("isi_bin", lambda x: sorted(x.dropna().unique())[:20]),
        )
    )
    missing_isi_bin = timing_data["isi_bin"].isna() & timing_data["isi"].notna()

    print("Filling missing isi_bin from raw isi:")
    print(timing_data.loc[missing_isi_bin, "participant_id"].value_counts())

    timing_data.loc[missing_isi_bin, "isi_bin"] = timing_data.loc[missing_isi_bin, "isi"]

    print("Timing rows with NaN isi_bin:")
    print(
        timing_data
        .groupby("participant_id")["isi_bin"]
        .apply(lambda x: x.isna().sum())
    )
    print("Participants in timing_data:")
    print(timing_data["participant_id"].unique())

    print("Participants in signal_data:")
    print(signal_data["participant_id"].unique())

    print("Timing counts:")
    print(timing_data["participant_id"].value_counts())

    print("Signal counts:")
    print(signal_data["participant_id"].value_counts())
    print("Preprocessing EMG...")
    processed_tables = []

    for participant_id, dfp in signal_data.groupby("participant_id", sort=False):
        print(f"Processing {participant_id}")
        dfp = dfp.sort_values("timestamp").copy()
        dfp = preprocess_emg_signal_table(dfp, C)
        processed_tables.append(dfp)

    signal_data = pd.concat(processed_tables, ignore_index=True)

    print("Extracting EMG reaction times...")
    rt_data = extract_emg_reaction_times(
        signal_data=signal_data,
        timing_data=timing_data,
        emg_var=C["emg_patterns"]["preprocess"]["output_var"],
        baseline_window_s=(-0.5, -0.1),
        response_window_s=(0.0, 1.5),
        threshold_sd=2.0,
        smooth_window_s=0.05,
        onset_fraction=0.20,
        min_peak_prominence_sd=3.0,
        rectify=True,
    )

    rt_data = add_reaction_time_normalization(
        rt_data,
        value_col="reaction_time_ms",
        group_cols=("participant_id", "isi_bin"),
        n_baseline_trials=10,
    )

    for participant_id in rt_data["participant_id"].unique():

        quick_plot_rt_detection(
            signal_data=signal_data,
            timing_data=timing_data,
            rt_data=rt_data,
            participant_id=participant_id,
            trial_num=np.random.randint(1, 401), #random number between 1 and 400
            emg_var=C["emg_patterns"]["preprocess"]["output_var"],
        )

    print(rt_data[["participant_id", "isi_bin", "trial_num", "reaction_time_ms", "reaction_time_valid", "emg_onset_threshold"]].head(20))

    print(rt_data.groupby(["participant_id", "isi_bin"])["reaction_time_valid"].sum())
    print(rt_data.groupby(["participant_id", "isi_bin"])["reaction_time_ms"].describe())

    rt_data.to_csv(TABLES_DIR / "reaction_time_metrics.csv", index=False)

    print("Saved:")
    print(TABLES_DIR / "reaction_time_metrics.csv")

    print("Valid reaction times:")
    print(rt_data.groupby(["participant_id", "isi_bin"])["reaction_time_valid"].sum())

    print("Plotting reaction time over trials by ISI...")
    plot_reaction_time_one_figure_per_participant(
        rt_data=rt_data,
        plots_dir=PLOTS_DIR,
        subject_colors=subject_colors,
        y_col="reaction_time_norm_delta",
        y_label="Reaction time change from first 10 trials (ms)",
        block_size=80,
        fig_prefix="reaction_time_change_by_isi_over_global_trials",
        x_col="trial_num",
        x_label="Global trial number",
        xlim=(0.5, 400.5),
    )
    print("Plotting raw reaction time over trials by ISI...")
    plot_reaction_time_one_figure_per_participant(
        rt_data=rt_data,
        plots_dir=PLOTS_DIR,
        subject_colors=subject_colors,
        y_col="reaction_time_ms",
        y_label="Reaction time (ms)",
        block_size=80,
        fig_prefix="reaction_time_raw_by_isi_over_global_trials",
        x_col="trial_num",
        x_label="Global trial number",
        xlim=(0.5, 400.5),
    )
    print("Plotting reaction time variability over trials by ISI...")
    plot_reaction_time_variability_one_figure_per_participant(
        rt_data=rt_data,
        plots_dir=PLOTS_DIR,
        subject_colors=subject_colors,
        rt_col="reaction_time_ms",
        rolling_window=5,
        min_periods=5,
        block_size=80,
        fig_prefix="reaction_time_variability_by_isi_over_global_trials",
        x_col="trial_num",
        x_label="Global trial number",
        xlim=(0.5, 400.5),
    )

    print("Done. Plots saved to:")
    print(PLOTS_DIR)

if __name__ == "__main__":
    main()