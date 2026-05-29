import numpy as np


def add_isi_bin_column(timing_df, config):
    """
    Assign each trial to one of the expected experimental ISI values.
    """

    timing_df = timing_df.copy()

    profile_cfg = config["emg_patterns"]["profile"]

    expected_isi = np.asarray(profile_cfg["expected_isi_values"], dtype=float)
    tolerance = profile_cfg.get("isi_tolerance", 0.05)

    isi_values = timing_df["isi"].astype(float).to_numpy()
    isi_bin = np.full(len(isi_values), np.nan)

    for i, isi in enumerate(isi_values):
        distances = np.abs(expected_isi - isi)
        closest_idx = np.argmin(distances)

        if distances[closest_idx] <= tolerance:
            isi_bin[i] = expected_isi[closest_idx]

    timing_df["isi_bin"] = isi_bin

    return timing_df