import numpy as np
import pandas as pd


def smooth_moving_average(y, fs, window_s=0.05):
    """
    Smooth signal with a moving average window.

    Example:
        window_s = 0.05 means 50 ms smoothing.
    """

    y = np.asarray(y, dtype=float)

    if not np.isfinite(fs) or fs <= 0:
        return y

    window_samples = int(round(window_s * fs))
    window_samples = max(1, window_samples)

    if window_samples <= 1:
        return y

    kernel = np.ones(window_samples) / window_samples

    y_filled = y.copy()
    valid = np.isfinite(y_filled)

    if valid.sum() < 2:
        return y

    if not np.all(valid):
        idx = np.arange(len(y_filled))
        y_filled[~valid] = np.interp(idx[~valid], idx[valid], y_filled[valid])

    return np.convolve(y_filled, kernel, mode="same")


def estimate_sampling_frequency(timestamps):
    timestamps = np.asarray(timestamps, dtype=float)
    dt = np.diff(timestamps)
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(dt) == 0:
        return np.nan

    return 1.0 / np.median(dt)

def detect_emg_onset_threshold(
    timestamps,
    emg,
    event_time,
    baseline_window_s=(-0.5, -0.1),
    response_window_s=(0.0, 1.5),
    threshold_sd=2.0,
    smooth_window_s=0.05,
    onset_fraction=0.20,
    min_peak_prominence_sd=3.0,
    rectify=True,
):
    """
    Detect EMG onset after an event using a smoothed EMG envelope.

    Detection logic:
        1. Rectify EMG if requested.
        2. Smooth the rectified signal to obtain an envelope.
        3. Estimate baseline from the pre-event window.
        4. Find the response peak after the event.
        5. Validate that the response peak is sufficiently above baseline.
        6. Define onset as the first time the envelope reaches a fraction
           of the response amplitude above baseline.

    This is less sensitive to isolated threshold crossings than simple
    first-threshold-crossing detection.
    """

    timestamps = np.asarray(timestamps, dtype=float)
    emg = np.asarray(emg, dtype=float)

    fs = estimate_sampling_frequency(timestamps)

    if rectify:
        emg_for_detection = np.abs(emg)
    else:
        emg_for_detection = emg.copy()

    emg_env = smooth_moving_average(
        emg_for_detection,
        fs=fs,
        window_s=smooth_window_s,
    )

    baseline_start = event_time + baseline_window_s[0]
    baseline_end = event_time + baseline_window_s[1]

    response_start = event_time + response_window_s[0]
    response_end = event_time + response_window_s[1]

    baseline_mask = (timestamps >= baseline_start) & (timestamps < baseline_end)
    response_mask = (timestamps >= response_start) & (timestamps < response_end)

    if baseline_mask.sum() < 5 or response_mask.sum() < 5:
        return np.nan, np.nan, np.nan, False

    baseline_env = emg_env[baseline_mask]
    response_env = emg_env[response_mask]
    response_time = timestamps[response_mask]

    baseline_env = baseline_env[np.isfinite(baseline_env)]
    valid_response = np.isfinite(response_env)

    if len(baseline_env) < 5 or valid_response.sum() < 5:
        return np.nan, np.nan, np.nan, False

    baseline_mean = np.nanmean(baseline_env)
    baseline_sd = np.nanstd(baseline_env)

    if not np.isfinite(baseline_mean) or not np.isfinite(baseline_sd):
        return np.nan, np.nan, np.nan, False

    response_peak = np.nanmax(response_env)

    # Trial is invalid if no clear response above baseline
    min_peak_threshold = baseline_mean + min_peak_prominence_sd * baseline_sd

    if not np.isfinite(response_peak) or response_peak < min_peak_threshold:
        return np.nan, np.nan, min_peak_threshold, False

    # Adaptive onset threshold based on response amplitude
    onset_threshold = baseline_mean + onset_fraction * (response_peak - baseline_mean)

    above = response_env >= onset_threshold

    if not np.any(above):
        return np.nan, np.nan, onset_threshold, False

    first_idx = np.where(above)[0][0]
    onset_time = response_time[first_idx]
    reaction_time_s = onset_time - event_time

    return onset_time, reaction_time_s, onset_threshold, True

def extract_emg_reaction_times(
    signal_data,
    timing_data,
    emg_var="emg_processed",
    baseline_window_s=(-0.5, -0.1),
    response_window_s=(0.0, 1.5),
    threshold_sd=2.0,
    smooth_window_s=0.05,
    onset_fraction=0.20,
    min_peak_prominence_sd=3.0,
    rectify=True,
):
    """
    Extract EMG reaction time for each trial.

    TDR / reaction time:
        EMG onset time - event time
    """

    rows = []

    for participant_id in timing_data["participant_id"].dropna().unique():

        signal_p = signal_data[
            signal_data["participant_id"] == participant_id
        ].sort_values("timestamp").reset_index(drop=True)

        timing_p = timing_data[
            timing_data["participant_id"] == participant_id
        ].sort_values("event").reset_index(drop=True)

        if len(signal_p) == 0 or len(timing_p) == 0:
            continue

        timestamps = signal_p["timestamp"].to_numpy(dtype=float)

        if emg_var not in signal_p.columns:
            raise KeyError(f"{emg_var} not found in signal_data.")

        emg = signal_p[emg_var].to_numpy(dtype=float)

        for i in range(len(timing_p)):

            event_time = timing_p.loc[i, "event"]

            if not np.isfinite(event_time):
                continue

            onset_time, reaction_time_s, threshold, is_valid = detect_emg_onset_threshold(
                timestamps=timestamps,
                emg=emg,
                event_time=event_time,
                baseline_window_s=baseline_window_s,
                response_window_s=response_window_s,
                threshold_sd=threshold_sd,
                smooth_window_s=smooth_window_s,
                onset_fraction=onset_fraction,
                min_peak_prominence_sd=min_peak_prominence_sd,
                rectify=rectify,
            )

            row = {
                "participant_id": participant_id,
                "trial_num": timing_p.loc[i, "trial_num"] if "trial_num" in timing_p.columns else i + 1,
                "isi": timing_p.loc[i, "isi"] if "isi" in timing_p.columns else np.nan,
                "isi_bin": timing_p.loc[i, "isi_bin"] if "isi_bin" in timing_p.columns else np.nan,
                "event": event_time,
                "emg_onset_time": onset_time,
                "reaction_time_s": reaction_time_s,
                "reaction_time_ms": reaction_time_s * 1000 if np.isfinite(reaction_time_s) else np.nan,
                "emg_onset_threshold": threshold,
                "reaction_time_valid": is_valid,
            }

            rows.append(row)

    rt_data = pd.DataFrame(rows)

    if rt_data.empty:
        return rt_data

    # Trial index within each ISI, so each ISI has x = 1..100
    rt_data = rt_data.sort_values(["participant_id", "isi_bin", "trial_num"]).reset_index(drop=True)

    rt_data["trial_within_isi"] = (
        rt_data
        .groupby(["participant_id", "isi_bin"])
        .cumcount()
        + 1
    )

    return rt_data


def add_reaction_time_normalization(
    rt_data,
    value_col="reaction_time_ms",
    group_cols=("participant_id", "isi_bin"),
    n_baseline_trials=10,
):
    """
    Normalize reaction time by the first N valid trials within each participant/ISI.

    Adds:
        reaction_time_baseline
        reaction_time_norm_delta
        reaction_time_norm_ratio
    """

    rt_data = rt_data.copy()

    rt_data["reaction_time_baseline"] = np.nan

    for keys, df_group in rt_data.groupby(list(group_cols)):

        valid = df_group[
            df_group["reaction_time_valid"]
            & np.isfinite(df_group[value_col])
        ].sort_values("trial_within_isi")

        baseline_values = valid[value_col].head(n_baseline_trials)

        if len(baseline_values) == 0:
            continue

        baseline = baseline_values.mean()

        idx = df_group.index

        rt_data.loc[idx, "reaction_time_baseline"] = baseline

    rt_data["reaction_time_norm_delta"] = (
        rt_data[value_col] - rt_data["reaction_time_baseline"]
    )

    rt_data["reaction_time_norm_ratio"] = (
        rt_data[value_col] / rt_data["reaction_time_baseline"]
    )

    return rt_data

def prepare_stim_nostim_rt_comparison(
    rt_data,
    rt_col="reaction_time_ms",
    n_baseline_trials=10,
):
    """
    Prepare paired STIM versus NOSTIM reaction-time data.

    For each session and ISI:
        centered RT = RT - mean of the first n valid RTs

    STIM and NOSTIM are then paired using:
        base participant + isi_bin + global trial_num
    """

    df = rt_data.copy()

    required_cols = {
        "participant_id",
        "trial_num",
        "isi_bin",
        "reaction_time_valid",
        rt_col,
    }

    missing_cols = required_cols.difference(df.columns)

    if missing_cols:
        raise KeyError(
            f"Missing required columns: {sorted(missing_cols)}"
        )

    # Extract participant name and experimental condition
    df["condition"] = np.select(
        [
            df["participant_id"].str.endswith("_NOSTIM"),
            df["participant_id"].str.endswith("_STIM"),
        ],
        [
            "NOSTIM",
            "STIM",
        ],
        default=np.nan,
    )

    df["base_participant"] = (
        df["participant_id"]
        .str.replace("_NOSTIM", "", regex=False)
        .str.replace("_STIM", "", regex=False)
    )

    # Keep only STIM/NOSTIM sessions
    df = df[df["condition"].isin(["STIM", "NOSTIM"])].copy()

    # Invalid RTs should not contribute to baseline or comparison
    df["rt_valid_value"] = df[rt_col].where(
        df["reaction_time_valid"]
        & np.isfinite(df[rt_col])
    )

    # Compute baseline from the first n valid trials of each session and ISI
    baseline_rows = (
        df[df["rt_valid_value"].notna()]
        .sort_values(
            [
                "base_participant",
                "condition",
                "isi_bin",
                "trial_num",
            ]
        )
        .groupby(
            [
                "base_participant",
                "condition",
                "isi_bin",
            ],
            group_keys=False,
        )
        .head(n_baseline_trials)
    )

    baselines = (
        baseline_rows
        .groupby(
            [
                "base_participant",
                "condition",
                "isi_bin",
            ]
        )["rt_valid_value"]
        .mean()
        .rename("rt_baseline_ms")
        .reset_index()
    )

    df = df.merge(
        baselines,
        on=[
            "base_participant",
            "condition",
            "isi_bin",
        ],
        how="left",
    )

    df["reaction_time_centered_ms"] = (
        df["rt_valid_value"] - df["rt_baseline_ms"]
    )

    # Put STIM and NOSTIM side by side
    paired = df.pivot_table(
        index=[
            "base_participant",
            "trial_num",
            "isi_bin",
        ],
        columns="condition",
        values=[
            rt_col,
            "reaction_time_centered_ms",
            "reaction_time_valid",
        ],
        aggfunc="first",
    )

    paired.columns = [
        f"{variable}_{condition.lower()}"
        for variable, condition in paired.columns
    ]

    paired = paired.reset_index()

    # Paired differences
    paired["rt_difference_raw_ms"] = (
        paired[f"{rt_col}_stim"]
        - paired[f"{rt_col}_nostim"]
    )

    paired["rt_difference_centered_ms"] = (
        paired["reaction_time_centered_ms_stim"]
        - paired["reaction_time_centered_ms_nostim"]
    )

    paired["pair_valid"] = (
        paired["reaction_time_valid_stim"].fillna(False).astype(bool)
        & paired["reaction_time_valid_nostim"].fillna(False).astype(bool)
        & np.isfinite(paired["reaction_time_centered_ms_stim"])
        & np.isfinite(paired["reaction_time_centered_ms_nostim"])
    )

    return df, paired
