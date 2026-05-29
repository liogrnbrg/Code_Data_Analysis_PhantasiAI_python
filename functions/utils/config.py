import numpy as np


def get_config():
    C = {}

    C["data"] = {
        "path": "/Users/liogr/Library/CloudStorage/OneDrive-Bibliothèquespartagées-UniversitedeMontreal/Yosra Cherni - Lio Grienenberger - MSc epfl/Codes/PhantasiAI_Python/data",
    }

    C["plot"] = {
        "subject_colors": {
            "Lio": np.array([0.00, 0.45, 0.74]),
            "Parisa": np.array([0.85, 0.33, 0.10]),
            "Mara": np.array([0.47, 0.67, 0.19]),
        },
        "font": {
            "family": "Arial", # Options: 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace', 'normal' ...
            "size": 12,
        },
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

    return C