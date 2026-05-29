import numpy as np
from scipy.signal import butter, filtfilt


def estimate_sampling_frequency(timestamp):
    """
    Estimate sampling frequency from timestamp vector.
    Uses unique timestamps because your file can contain duplicate timestamps.
    """
    t = np.asarray(timestamp, dtype=float)
    t = np.unique(t)

    if len(t) < 3:
        raise ValueError("Not enough timestamps to estimate sampling frequency.")

    dt = np.nanmedian(np.diff(t))

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Could not estimate sampling frequency from timestamp.")

    fs = 1.0 / dt
    return fs


def apply_filter(y, fs, filter_type, cutoff_hz, order=4):
    """
    Generic Butterworth zero-phase filtering.
    filter_type: 'bandpass' or 'lowpass'
    """

    nyquist = fs / 2.0

    if filter_type == "bandpass":
        low, high = cutoff_hz

        if high >= nyquist:
            print(
                f"Warning: high cutoff {high:.1f} Hz >= Nyquist {nyquist:.1f} Hz. "
                f"Reducing high cutoff to {0.90 * nyquist:.1f} Hz."
            )
            high = 0.90 * nyquist

        if low >= high:
            print(
                f"Warning: bandpass skipped because low cutoff {low:.1f} Hz "
                f">= high cutoff {high:.1f} Hz."
            )
            return y

        wn = [low / nyquist, high / nyquist]
        b, a = butter(order, wn, btype="bandpass")

    elif filter_type == "lowpass":
        cutoff = cutoff_hz

        if cutoff >= nyquist:
            print(
                f"Warning: lowpass skipped because cutoff {cutoff:.1f} Hz "
                f">= Nyquist {nyquist:.1f} Hz."
            )
            return y

        wn = cutoff / nyquist
        b, a = butter(order, wn, btype="lowpass")

    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    return filtfilt(b, a, y)


def preprocess_emg_signal_table(df, config):
    """
    Preprocess EMG signal inside a pandas DataFrame.

    Adds a new column, for example:
        emg_processed

    Expected config structure:
        config["emg_patterns"]["preprocess"]
    """

    cfg = config["emg_patterns"]["preprocess"]

    input_var = cfg["input_var"]
    output_var = cfg["output_var"]

    if input_var not in df.columns:
        raise KeyError(f'Input EMG column "{input_var}" not found in dataframe.')

    if "timestamp" not in df.columns:
        raise KeyError('Column "timestamp" not found in dataframe.')

    df = df.copy()

    y = df[input_var].astype(float).to_numpy()

    # If disabled, just copy raw signal to output column
    if not cfg["enabled"]:
        df[output_var] = y
        return df

    fs = estimate_sampling_frequency(df["timestamp"].to_numpy())

    print(f"Estimated sampling frequency: {fs:.2f} Hz")

    # Remove DC / mean offset
    if cfg["remove_dc"]:
        y = y - np.nanmean(y)

    # Bandpass
    if cfg["bandpass_enabled"]:
        y = apply_filter(
            y=y,
            fs=fs,
            filter_type="bandpass",
            cutoff_hz=cfg["bandpass_range_hz"],
            order=cfg["bandpass_order"],
        )

    # Rectification
    if cfg["rectify"]:
        y = np.abs(y)

    # Envelope
    if cfg["envelope_enabled"]:
        y = apply_filter(
            y=y,
            fs=fs,
            filter_type="lowpass",
            cutoff_hz=cfg["envelope_lowpass_hz"],
            order=cfg["envelope_order"],
        )

    df[output_var] = y

    return df