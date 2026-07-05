# config.py

import numpy as np


def get_config():
    C = {}

    C["data"] = {
        "path": "/Users/liogr/Library/CloudStorage/OneDrive-Bibliothèquespartagées-UniversitedeMontreal/Yosra Cherni - Lio Grienenberger - MSc epfl/Codes/PhantasiAI_Python/data",
    }

    C["plot"] = {
        "subject_colors": {
            "Lio_STIM": np.array([0, 140, 255]) / 255,
            "Lio_STIM_2": np.array([17, 0, 255]) / 255,
            "Lio_NOSTIM": np.array([221, 0, 255]) / 255,
            "Parisa_STIM": np.array([0.85, 0.33, 0.10]),
            "Parisa_NOSTIM": np.array([255, 0, 85]) / 255,
            "Mara_STIM": np.array([0.47, 0.67, 0.19]),
            "Mara_NOSTIM": np.array([0.47, 0.67, 0.19]) * 0.5,
        },
        "font": {
            "family": "DejaVu Sans", # Options: 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace', 'normal' ...
            "size": 12,
        },
    }

    C["conditions"] = {
        "condition_order": [
            "NOSTIM",
            "FIXED_STIM",
            "STIM",
        ],

        "condition_labels": {
            "NOSTIM": "No stimulation",
            "FIXED_STIM": "Fixed stimulation",
            "STIM": "Optimized stimulation",
        },

        "comparison_pairs": [
            {
                "comparison_name": "stim_vs_nostim",
                "test_condition": "STIM",
                "reference_condition": "NOSTIM",
                "comparison_label": "Optimized stimulation − no stimulation",
            },
            {
                "comparison_name": "fixed_stim_vs_nostim",
                "test_condition": "FIXED_STIM",
                "reference_condition": "NOSTIM",
                "comparison_label": "Fixed stimulation − no stimulation",
            },
            {
                "comparison_name": "stim_vs_fixed_stim",
                "test_condition": "STIM",
                "reference_condition": "FIXED_STIM",
                "comparison_label": "Optimized stimulation − fixed stimulation",
            },
        ],
    }
    C

    C["emg_patterns"] = {}

    C["emg_patterns"]["preprocess"] = {
        "enabled": True,
        "input_var": "ch1 (µV)",
        "output_var": "emg_processed",

        "remove_dc": True,

        # Important: leave bandpass disabled for now because your timestamps
        # suggest a relatively low sampling frequency.
        "bandpass_enabled": False,
        "bandpass_range_hz": [20, 450],
        "bandpass_order": 4,

        "rectify": True,

        "envelope_enabled": False,
        "envelope_lowpass_hz": 10,
        "envelope_order": 4,
    }

    C["emg_patterns"]["profile"] = {
        "n_points": 200,
        "time_mode": "seconds",
        "duration_strategy": "median_duration",
        "max_trials_per_isi": None,  # None later when everything is checked

        # Current experimental ISIs
        "expected_isi_values": [2.0, 2.5, 3.0, 3.5],
        "isi_tolerance": 0.05,
    }

    C["emg_patterns"]["plot"] = {
        "face_alpha": 0.16,
        "line_width": 2.2,
        "emg_ylabel": "Rectified EMG amplitude (µV)",
        "accel_ylabel": "Acceleration",
    }

    C["accel_preprocess"] = {
        "enabled": True,
        "input_cols": ["accel_x", "accel_y", "accel_z"],
        "output_suffix": "_preprocessed",

        "remove_dc": False,
        "center_method": "median",

        "lowpass": {
            "enabled": True,
            "cutoff_hz": 3.0,
            "order": 4,
        },

        "artifact_rejection": {
            "enabled": False,
        },
    }

    C["reaction_time"] = {
        "emg_var": "emg_processed",

        "baseline_window_s": [-0.5, -0.1],
        "response_window_s": [0.0, 1.5],

        "threshold_sd": 3.0,
        "min_duration_s": 0.03,

        "n_baseline_trials_normalization": 10,
        "block_size": 40,
    }
    
    C["emg_features"] = {
        "raw_emg_var": "ch1 (µV)",
        "processed_emg_var": "emg_processed",

        # Each trial = current event to next event
        "trial_window": {
            "mode": "event_to_next_event",
            "start_offset_s": 0.0,
            "end_offset_s": 0.0,
        },

        # Pre-event baseline used to estimate resting EMG before each cue
        "pre_event_baseline_window_s": [-0.5, -0.1],

        "min_samples_per_window": 20,

        # Normalize each session to its first trials
        "session_baseline": {
            "n_trials": 10,
            "stat": "median",
        },

        "frequency": {
            "enabled": True,
            "min_freq_hz": 5.0,

            # Important because some sessions are sampled around 72–76 Hz.
            # Nyquist is therefore around 36–38 Hz.
            "max_freq_hz": 35.0,

            "welch_nperseg": 256,
        },

        "plot": {
            "block_size": 40,
            "band": "sd",
            "sd_multiplier": 1.0,
            "ci_level": 0.95,
            "alpha": 0.05,

            # Main combined plots: all sessions on one figure
            "make_combined_regression_plots": True,
            "make_combined_block_plots": True,

            # Individual Lio/Parisa-separated plots
            "make_individual_plots": False,

            # One combined figure per ISI
            "make_split_by_isi_plots": True,

            "show_individual_points": True,
        },

                "normalization": {
            # Options: "zscore", "percent", "centered", "raw"
            "plot_method": "zscore",
        },

        "variability": {
            "enabled": True,
            "rolling_window_trials": 10,
            "rolling_min_periods": 5,
            "features": [
                "rms_response",
                "rms_response_minus_pre_event",
            ],
        },
    }


    return C