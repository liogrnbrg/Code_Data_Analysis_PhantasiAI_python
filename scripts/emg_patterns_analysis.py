from pathlib import Path
import sys
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
FUNCTIONS_DIR = PROJECT_DIR / "functions"

sys.path.append(str(FUNCTIONS_DIR))

from utils.config import get_config

import utils.config as config_module

print("Config file used:", config_module.__file__)

from loading.load_data import load_timing_data, load_emg_accel_data
from preprocessing.isi_binning import add_isi_bin_column
from preprocessing.preprocess_emg import preprocess_emg_signal_table
from preprocessing.signal_segments import select_signal_segment
from plotting.plot_segments import (
    plot_emg_segment_by_participant,
    plot_accel_segment_by_participant,
)
from plotting.plot_activation_profiles import (
    plot_activation_profiles_grid_by_participant_and_isi,
)

C = get_config()

DATA_DIR = Path(C["data"]["path"]).expanduser()

if not DATA_DIR.exists():
    DATA_DIR = PROJECT_DIR / "data"

if not DATA_DIR.exists():
    raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

PLOTS_DIR = PROJECT_DIR / "outputs" / "plots" / "emg_patterns"

def main():
    C = get_config()
    subject_colors = C["plot"]["subject_colors"]

    fraction_to_plot = 1 / 50
    segment_position = "start"  # "start", "middle", or "end"

    print("Loading data...")
    timing_data = load_timing_data(DATA_DIR)
    emg_data = load_emg_accel_data(DATA_DIR)

    timing_data = add_isi_bin_column(timing_data, C)
    
    print("ISI bins found:")
    print(timing_data.groupby("participant_id")["isi_bin"].value_counts(dropna=False))

    print("Preprocessing EMG...")
    processed_tables = []

    for participant_id, dfp in emg_data.groupby("participant_id", sort=False):
        print(f"Processing {participant_id}")

        dfp = dfp.sort_values("timestamp").copy()
        dfp = preprocess_emg_signal_table(dfp, C)

        processed_tables.append(dfp)

    emg_data_prep = pd.concat(processed_tables, ignore_index=True)

    print("Plotting simple EMG and acceleration segments...")

    participants = emg_data_prep["participant_id"].unique()

    for participant_id in participants:
        data_p = emg_data_prep[emg_data_prep["participant_id"] == participant_id].copy()
        data_p = data_p.sort_values("timestamp")

        timing_p = timing_data[timing_data["participant_id"] == participant_id].copy()
        timing_p = timing_p.sort_values("event")

        segment_p = select_signal_segment(
            data_p,
            fraction_to_plot=fraction_to_plot,
            segment_position=segment_position,
        )

        plot_emg_segment_by_participant(
            segment_df=segment_p,
            timing_df=timing_p,
            participant_id=participant_id,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
            signal_var=C["emg_patterns"]["preprocess"]["output_var"],
            y_label=C["emg_patterns"]["plot"]["emg_ylabel"],
            fig_suffix="processed_EMG_segment",
        )

        plot_accel_segment_by_participant(
            segment_df=segment_p,
            participant_id=participant_id,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
            fig_suffix="acceleration_segment",
        )

    print("Plotting activation profiles...")

    participants_profiles = [
        p for p in emg_data_prep["participant_id"].unique()
        if p in timing_data["participant_id"].unique()
    ]

    plot_activation_profiles_grid_by_participant_and_isi(
        signal_data=emg_data_prep,
        timing_data=timing_data,
        participants=participants_profiles,
        signal_var=C["emg_patterns"]["preprocess"]["output_var"],
        y_label=C["emg_patterns"]["plot"]["emg_ylabel"],
        fig_title="Average ± SD processed EMG profiles from current event to next event",
        fig_name="processed_emg_profiles_trial_to_next_trial_grid.png",
        plots_dir=PLOTS_DIR,
        subject_colors=subject_colors,
        config=C,
    )

    plot_activation_profiles_grid_by_participant_and_isi(
        signal_data=emg_data_prep,
        timing_data=timing_data,
        participants=participants_profiles,
        signal_var="accel_x",
        y_label="Acceleration X",
        fig_title="Average ± SD acceleration X profiles from current event to next event",
        fig_name="accel_x_profiles_trial_to_next_trial_grid.png",
        plots_dir=PLOTS_DIR,
        subject_colors=subject_colors,
        config=C,
    )

    plot_activation_profiles_grid_by_participant_and_isi(
        signal_data=emg_data_prep,
        timing_data=timing_data,
        participants=participants_profiles,
        signal_var="accel_y",
        y_label="Acceleration Y",
        fig_title="Average ± SD acceleration Y profiles from current event to next event",
        fig_name="accel_y_profiles_trial_to_next_trial_grid.png",
        plots_dir=PLOTS_DIR,
        subject_colors=subject_colors,
        config=C,
    )

    plot_activation_profiles_grid_by_participant_and_isi(
        signal_data=emg_data_prep,
        timing_data=timing_data,
        participants=participants_profiles,
        signal_var="accel_z",
        y_label="Acceleration Z",
        fig_title="Average ± SD acceleration Z profiles from current event to next event",
        fig_name="accel_z_profiles_trial_to_next_trial_grid.png",
        plots_dir=PLOTS_DIR,
        subject_colors=subject_colors,
        config=C,
    )

    print("Done. Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()