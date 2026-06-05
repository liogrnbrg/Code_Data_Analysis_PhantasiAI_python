import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def estimate_sampling_frequency(timestamps):
    """
    Estimate sampling frequency from timestamps using the median time step.
    """

    timestamps = np.asarray(timestamps, dtype=float)
    dt = np.diff(timestamps)
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(dt) == 0:
        return np.nan

    return 1.0 / np.median(dt)


def fill_missing_by_interpolation(y):
    """
    Fill NaNs by linear interpolation.
    """

    y = np.asarray(y, dtype=float)

    if np.all(np.isfinite(y)):
        return y

    valid = np.isfinite(y)

    if valid.sum() < 2:
        return y

    idx = np.arange(len(y))
    y_filled = y.copy()
    y_filled[~valid] = np.interp(idx[~valid], idx[valid], y[valid])

    return y_filled


def lowpass_filter_signal(y, fs, cutoff_hz, order=4):
    """
    Apply zero-phase Butterworth low-pass filter.
    """

    y = np.asarray(y, dtype=float)

    if not np.isfinite(fs) or fs <= 0:
        print("Warning: invalid sampling frequency. Skipping low-pass filter.")
        return y

    nyquist = fs / 2

    if cutoff_hz >= nyquist:
        print(
            f"Warning: low-pass cutoff {cutoff_hz} Hz >= Nyquist {nyquist:.2f} Hz. "
            "Skipping filter."
        )
        return y

    y_filled = fill_missing_by_interpolation(y)

    b, a = butter(order, cutoff_hz / nyquist, btype="low")

    min_len = 3 * max(len(a), len(b))

    if len(y_filled) <= min_len:
        print("Warning: signal too short for filtfilt. Skipping filter.")
        return y

    return filtfilt(b, a, y_filled)


def preprocess_accel_signal_table(data, config):
    """
    Preprocess acceleration signals.

    Steps are controlled by config["accel_preprocess"].

    Current steps:
        1. optional DC removal / centering
        2. optional low-pass filtering

    Output columns are saved as:
        accel_x_preprocessed
        accel_y_preprocessed
        accel_z_preprocessed

    Raw columns are preserved.
    """

    data = data.sort_values("timestamp").copy()

    if "accel_preprocess" not in config:
        raise KeyError("Missing config['accel_preprocess'].")

    C = config["accel_preprocess"]

    input_cols = C.get("input_cols", ["accel_x", "accel_y", "accel_z"])
    output_suffix = C.get("output_suffix", "_preprocessed")

    fs = estimate_sampling_frequency(data["timestamp"].to_numpy(dtype=float))
    print(f"Estimated acceleration sampling frequency: {fs:.2f} Hz")

    for col in input_cols:

        if col not in data.columns:
            print(f"Warning: {col} not found. Skipping.")
            continue

        y = data[col].to_numpy(dtype=float)

        # ---------- 1. Remove DC / center ----------
        if C.get("remove_dc", False):
            center_method = C.get("center_method", "median")

            if center_method == "median":
                y = y - np.nanmedian(y)
            elif center_method == "mean":
                y = y - np.nanmean(y)
            else:
                raise ValueError(f"Unknown center_method: {center_method}")

        # ---------- 2. Low-pass filter ----------
        if C.get("lowpass", {}).get("enabled", False):
            y = lowpass_filter_signal(
                y,
                fs=fs,
                cutoff_hz=C["lowpass"].get("cutoff_hz", 10),
                order=C["lowpass"].get("order", 4),
            )

        # ---------- Future preprocessing steps can be added here ----------
        # Example:
        # if C.get("artifact_rejection", {}).get("enabled", False):
        #     y = ...

        output_col = f"{col}{output_suffix}"
        data[output_col] = y

    return data