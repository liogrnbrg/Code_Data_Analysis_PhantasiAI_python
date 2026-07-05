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
from plotting.plot_segments import plot_3axis_segment_by_participant
from plotting.plot_activation_profiles import (
    plot_activation_profiles_grid_by_participant_and_isi,
)
# from preprocessing.accel_integration import add_velocity_and_position_proxies, add_trialwise_velocity_position_proxies
from preprocessing.accel_features import add_trialwise_velocity_position_proxies
from preprocessing.preprocess_accel import preprocess_accel_signal_table

from plotting.plot_segments import (
    plot_emg_segment_by_participant,
    plot_accel_segment_by_participant,
    plot_speed_segment_by_participant,
    plot_position_segment_by_participant,
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

    fraction_to_plot = 1 / 80
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

        timing_p = timing_data[
            timing_data["participant_id"] == participant_id
        ].copy()

        # ---------- Preprocess acceleration ----------
        dfp = preprocess_accel_signal_table(dfp, C)

        # ---------- Velocity/position from raw acceleration ----------
        dfp = add_trialwise_velocity_position_proxies(
            data=dfp,
            timing_data=timing_p,
            accel_cols=("accel_x", "accel_y", "accel_z"),
            axis_names=("x_raw", "y_raw", "z_raw"),
            baseline_window_s=(-0.5, -0.1),
            force_zero_velocity_end=True,
            detrend_position=True,
        )

        # ---------- Velocity/position from preprocessed acceleration ----------
        dfp = add_trialwise_velocity_position_proxies(
            data=dfp,
            timing_data=timing_p,
            accel_cols=("accel_x_preprocessed", "accel_y_preprocessed", "accel_z_preprocessed"),
            axis_names=("x_preprocessed", "y_preprocessed", "z_preprocessed"),
            baseline_window_s=(-0.5, -0.1),
            force_zero_velocity_end=True,
            detrend_position=True,
        )

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

        # plot_accel_segment_by_participant(
        #     segment_df=segment_p,
        #     participant_id=participant_id,
        #     plots_dir=PLOTS_DIR,
        #     subject_colors=subject_colors,
        #     fig_suffix="acceleration_segment",
        # )

        # plot_speed_segment_by_participant(
        #     segment_df=segment_p,
        #     participant_id=participant_id,
        #     plots_dir=PLOTS_DIR,
        #     subject_colors=subject_colors,
        #     fig_suffix="velocity_segment",
        # )

        # plot_position_segment_by_participant(
        #     segment_df=segment_p,
        #     participant_id=participant_id,
        #     plots_dir=PLOTS_DIR,
        #     subject_colors=subject_colors,
        #     fig_suffix="position_segment",
        # )

        #
        plot_3axis_segment_by_participant(
            segment_df=segment_p,
            participant_id=participant_id,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
            signal_cols=("accel_x", "accel_y", "accel_z"),
            y_label="Raw acceleration",
            fig_suffix="acceleration_raw_segment",
        )

        # ---------- Plot all 3-axis signals in the segment ----------
        plot_3axis_segment_by_participant(
            segment_df=segment_p,
            participant_id=participant_id,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
            signal_cols=("accel_x_preprocessed", "accel_y_preprocessed", "accel_z_preprocessed"),
            y_label="Preprocessed acceleration",
            fig_suffix="acceleration_preprocessed_segment",
        )

        # ---------- Raw-derived velocity ----------
        plot_3axis_segment_by_participant(
            segment_df=segment_p,
            participant_id=participant_id,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
            signal_cols=("velocity_x_raw", "velocity_y_raw", "velocity_z_raw"),
            y_label="Velocity proxy from raw acceleration",
            fig_suffix="velocity_raw_segment",
        )

        # ---------- Preprocessed-derived velocity ----------
        plot_3axis_segment_by_participant(
            segment_df=segment_p,
            participant_id=participant_id,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
            signal_cols=("velocity_x_preprocessed", "velocity_y_preprocessed", "velocity_z_preprocessed"),
            y_label="Velocity proxy from preprocessed acceleration",
            fig_suffix="velocity_preprocessed_segment",
        )

        # ---------- Raw-derived position ----------
        plot_3axis_segment_by_participant(
            segment_df=segment_p,
            participant_id=participant_id,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
            signal_cols=("position_x_raw", "position_y_raw", "position_z_raw"),
            y_label="Position proxy from raw acceleration",
            fig_suffix="position_raw_segment",
        )

        # ---------- Preprocessed-derived position ----------
        plot_3axis_segment_by_participant(
            segment_df=segment_p,
            participant_id=participant_id,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
            signal_cols=("position_x_preprocessed", "position_y_preprocessed", "position_z_preprocessed"),
            y_label="Position proxy from preprocessed acceleration",
            fig_suffix="position_preprocessed_segment",
        )

    print("Plotting activation profiles...")

    participants_profiles = [
        p for p in emg_data_prep["participant_id"].unique()
        if p in timing_data["participant_id"].unique()
    ]

    emg_var = C["emg_patterns"]["preprocess"]["output_var"]

    plot_activation_profiles_grid_by_participant_and_isi(
        signal_data=emg_data_prep,
        timing_data=timing_data,
        participants=participants_profiles,
        signal_var=emg_var,
        y_label=C["emg_patterns"]["plot"]["emg_ylabel"],
        fig_title="Average ± SD EMG profiles from current event to next event",
        fig_name="emg_profiles_trial_to_next_trial_grid.png",
        plots_dir=PLOTS_DIR,
        subject_colors=subject_colors,
        config=C,
    )

    # for accel_var, y_label in [
    #     ("accel_x", "Acceleration X raw"),
    #     ("accel_y", "Acceleration Y raw"),
    #     ("accel_z", "Acceleration Z raw"),
    #     ("accel_x_preprocessed", "Acceleration X preprocessed"),
    #     ("accel_y_preprocessed", "Acceleration Y preprocessed"),
    #     ("accel_z_preprocessed", "Acceleration Z preprocessed"),
    # ]:
    #     plot_activation_profiles_grid_by_participant_and_isi(
    #         signal_data=emg_data_prep,
    #         timing_data=timing_data,
    #         participants=participants_profiles,
    #         signal_var=accel_var,
    #         y_label=y_label,
    #         fig_title=f"Average ± SD {y_label.lower()} profiles from current event to next event",
    #         fig_name=f"{accel_var}_profiles_trial_to_next_trial_grid.png",
    #         plots_dir=PLOTS_DIR,
    #         subject_colors=subject_colors,
    #         config=C,
    #     )
        
    # for velocity_var, y_label in [
    #     ("velocity_x_raw", "Velocity proxy X raw"),
    #     ("velocity_y_raw", "Velocity proxy Y raw"),
    #     ("velocity_z_raw", "Velocity proxy Z raw"),
    #     ("velocity_x_preprocessed", "Velocity proxy X preprocessed"),
    #     ("velocity_y_preprocessed", "Velocity proxy Y preprocessed"),
    #     ("velocity_z_preprocessed", "Velocity proxy Z preprocessed"),
    # ]:
    #     plot_activation_profiles_grid_by_participant_and_isi(
    #         signal_data=emg_data_prep,
    #         timing_data=timing_data,
    #         participants=participants_profiles,
    #         signal_var=velocity_var,
    #         y_label=y_label,
    #         fig_title=f"Average ± SD {y_label.lower()} profiles from current event to next event",
    #         fig_name=f"{velocity_var}_profiles_trial_to_next_trial_grid.png",
    #         plots_dir=PLOTS_DIR,
    #         subject_colors=subject_colors,
    #         config=C,
    #     )
    
    # for position_var, y_label in [
    #     ("position_x_raw", "Position proxy X raw"),
    #     ("position_y_raw", "Position proxy Y raw"),
    #     ("position_z_raw", "Position proxy Z raw"),
    #     ("position_x_preprocessed", "Position proxy X preprocessed"),
    #     ("position_y_preprocessed", "Position proxy Y preprocessed"),
    #     ("position_z_preprocessed", "Position proxy Z preprocessed"),
    # ]:
    #     plot_activation_profiles_grid_by_participant_and_isi(
    #         signal_data=emg_data_prep,
    #         timing_data=timing_data,
    #         participants=participants_profiles,
    #         signal_var=position_var,
    #         y_label=y_label,
    #         fig_title=f"Average ± SD {y_label.lower()} profiles from current event to next event",
    #         fig_name=f"{position_var}_profiles_trial_to_next_trial_grid.png",
    #         plots_dir=PLOTS_DIR,
    #         subject_colors=subject_colors,
    #         config=C,
    #     )

    print("Done. Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()