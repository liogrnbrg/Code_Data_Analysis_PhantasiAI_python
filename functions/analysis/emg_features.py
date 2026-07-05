# emg_features.py

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy.signal import welch

from preprocessing.preprocess_emg import estimate_sampling_frequency


def get_base_participant(participant_id):
    """
    Examples:
        Lio_NOSTIM -> Lio
        Lio_STIM -> Lio
        Lio_STIM_2 -> Lio
        Parisa_NOSTIM -> Parisa
    """
    participant_id = str(participant_id)

    return re.sub(
        r"_(NOSTIM|STIM)(_\d+)?$",
        "",
        participant_id,
    )


def get_session_type(participant_id):
    participant_id = str(participant_id)

    if "_NOSTIM" in participant_id:
        return "NOSTIM"

    if "_STIM" in participant_id:
        return "STIM"

    return "UNKNOWN"


def _rms(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) == 0:
        return np.nan

    return np.sqrt(np.mean(y ** 2))


def _mav(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) == 0:
        return np.nan

    return np.mean(np.abs(y))


def _iemg(y, fs):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) == 0 or not np.isfinite(fs) or fs <= 0:
        return np.nan

    return np.sum(np.abs(y)) / fs


def _peak_abs(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) == 0:
        return np.nan

    return np.max(np.abs(y))


def _time_to_peak(y, t_rel):
    y = np.asarray(y, dtype=float)
    t_rel = np.asarray(t_rel, dtype=float)

    valid = np.isfinite(y) & np.isfinite(t_rel)

    if valid.sum() == 0:
        return np.nan

    y = y[valid]
    t_rel = t_rel[valid]

    idx = np.argmax(np.abs(y))

    return t_rel[idx]


def _spectral_features(
    y,
    fs,
    min_freq_hz=5.0,
    max_freq_hz=None,
    welch_nperseg=256,
):
    """
    Compute median frequency, mean frequency, and total spectral power.

    Important: use raw/unrectified EMG, ideally DC-centered.
    """

    out = {
        "mdf_response_hz": np.nan,
        "mnf_response_hz": np.nan,
        "spectral_power_response": np.nan,
    }

    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) < 8 or not np.isfinite(fs) or fs <= 0:
        return out

    y = y - np.nanmean(y)

    nperseg = min(int(welch_nperseg), len(y))

    if nperseg < 8:
        return out

    freqs, psd = welch(
        y,
        fs=fs,
        nperseg=nperseg,
    )

    if max_freq_hz is None:
        max_freq_hz = fs / 2.0

    keep = (
        np.isfinite(freqs)
        & np.isfinite(psd)
        & (freqs >= min_freq_hz)
        & (freqs <= max_freq_hz)
    )

    freqs = freqs[keep]
    psd = psd[keep]

    if len(freqs) == 0 or np.nansum(psd) <= 0:
        return out

    total_power = np.trapezoid(psd, freqs)

    cumulative_power = np.cumsum(psd)
    half_power = cumulative_power[-1] / 2.0

    mdf_idx = np.where(cumulative_power >= half_power)[0]

    if len(mdf_idx) > 0:
        mdf = freqs[mdf_idx[0]]
    else:
        mdf = np.nan

    mnf = np.sum(freqs * psd) / np.sum(psd)

    out["mdf_response_hz"] = mdf
    out["mnf_response_hz"] = mnf
    out["spectral_power_response"] = total_power

    return out


def _get_baseline_value(values, stat="median"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    if stat == "median":
        return np.nanmedian(values)

    if stat == "mean":
        return np.nanmean(values)

    raise ValueError(f"Unknown baseline stat: {stat}")


def add_first_trials_normalization(
    features,
    feature_cols,
    n_trials=10,
    stat="median",
):
    """
    Normalize each feature within each session using the first n valid trials.

    Adds:
        feature_baseline_first10
        feature_centered_first10
        feature_pct_first10
        feature_ratio_first10
        feature_zscore_first10

    Z-score is computed as:
        z = (value - mean(first n valid trials)) / sd(first n valid trials)
    """

    features = features.copy()

    for participant_id, idx in features.groupby("participant_id").groups.items():
        session_data = features.loc[idx].sort_values("trial_num")

        for feature_col in feature_cols:
            if feature_col not in features.columns:
                continue

            first_values = (
                session_data[feature_col]
                .dropna()
                .head(n_trials)
                .to_numpy(dtype=float)
            )

            baseline = _get_baseline_value(
                first_values,
                stat=stat,
            )

            if len(first_values) >= 2:
                baseline_mean = np.nanmean(first_values)
                baseline_sd = np.nanstd(first_values, ddof=1)
            else:
                baseline_mean = np.nan
                baseline_sd = np.nan

            baseline_col = f"{feature_col}_baseline_first{n_trials}"
            centered_col = f"{feature_col}_centered_first{n_trials}"
            pct_col = f"{feature_col}_pct_first{n_trials}"
            ratio_col = f"{feature_col}_ratio_first{n_trials}"
            zscore_col = f"{feature_col}_zscore_first{n_trials}"

            baseline_mean_col = f"{feature_col}_baseline_mean_first{n_trials}"
            baseline_sd_col = f"{feature_col}_baseline_sd_first{n_trials}"

            values = features.loc[idx, feature_col].astype(float)

            features.loc[idx, baseline_col] = baseline
            features.loc[idx, baseline_mean_col] = baseline_mean
            features.loc[idx, baseline_sd_col] = baseline_sd

            features.loc[idx, centered_col] = values - baseline

            if np.isfinite(baseline) and baseline != 0:
                features.loc[idx, pct_col] = 100 * (values - baseline) / baseline
                features.loc[idx, ratio_col] = values / baseline
            else:
                features.loc[idx, pct_col] = np.nan
                features.loc[idx, ratio_col] = np.nan

            if np.isfinite(baseline_sd) and baseline_sd > 0:
                features.loc[idx, zscore_col] = (
                    values - baseline_mean
                ) / baseline_sd
            else:
                features.loc[idx, zscore_col] = np.nan

    return features


def add_rolling_feature_variability(
    features,
    feature_cols,
    window_trials=10,
    min_periods=5,
):
    """
    Add rolling trial-to-trial variability features within each session.

    Example:
        rms_response_rolling_sd_10

    This captures local variability across trials, useful for checking
    consistency, learning, fatigue, or stimulation-related stabilization.
    """

    features = features.copy()

    for participant_id, idx in features.groupby("participant_id").groups.items():
        session_data = (
            features
            .loc[idx]
            .sort_values("trial_num")
            .copy()
        )

        ordered_idx = session_data.index

        for feature_col in feature_cols:
            if feature_col not in features.columns:
                continue

            values = session_data[feature_col].astype(float)

            rolling_sd = (
                values
                .rolling(
                    window=window_trials,
                    min_periods=min_periods,
                    center=True,
                )
                .std(ddof=1)
            )

            rolling_mean = (
                values
                .rolling(
                    window=window_trials,
                    min_periods=min_periods,
                    center=True,
                )
                .mean()
            )

            rolling_cv = 100 * rolling_sd / rolling_mean.abs()

            sd_col = f"{feature_col}_rolling_sd_{window_trials}"
            cv_col = f"{feature_col}_rolling_cv_{window_trials}_pct"

            features.loc[ordered_idx, sd_col] = rolling_sd.to_numpy(dtype=float)
            features.loc[ordered_idx, cv_col] = rolling_cv.to_numpy(dtype=float)

    return features

def compute_emg_trial_features(
    signal_data,
    timing_data,
    config,
):
    """
    Compute trial-level EMG features.

    Each trial window goes from current event to next event.

    Output:
        one row per participant/session/trial.
    """

    cfg = config["emg_features"]

    raw_emg_var = cfg["raw_emg_var"]
    processed_emg_var = cfg["processed_emg_var"]

    min_samples = cfg.get("min_samples_per_window", 20)

    baseline_window_s = cfg.get(
        "pre_event_baseline_window_s",
        [-0.5, -0.1],
    )

    freq_cfg = cfg.get("frequency", {})
    frequency_enabled = freq_cfg.get("enabled", True)

    rows = []

    for participant_id, signal_p in signal_data.groupby("participant_id", sort=False):
        timing_p = timing_data[
            timing_data["participant_id"] == participant_id
        ].copy()

        if timing_p.empty:
            print(f"Skipping {participant_id}: no timing data.")
            continue

        signal_p = signal_p.sort_values("timestamp").copy()
        timing_p = timing_p.sort_values("event").reset_index(drop=True)

        if raw_emg_var not in signal_p.columns:
            raise KeyError(
                f'Raw EMG column "{raw_emg_var}" not found for {participant_id}.'
            )

        if processed_emg_var not in signal_p.columns:
            raise KeyError(
                f'Processed EMG column "{processed_emg_var}" not found for {participant_id}.'
            )

        timestamps = signal_p["timestamp"].to_numpy(dtype=float)
        raw_emg = signal_p[raw_emg_var].to_numpy(dtype=float)
        processed_emg = signal_p[processed_emg_var].to_numpy(dtype=float)

        try:
            fs = estimate_sampling_frequency(timestamps)
        except ValueError:
            fs = np.nan

        isi_col = "isi_bin" if "isi_bin" in timing_p.columns else "isi"

        for pos in range(len(timing_p) - 1):
            event_start = timing_p.iloc[pos]["event"]
            event_end = timing_p.iloc[pos + 1]["event"]

            if (
                not np.isfinite(event_start)
                or not np.isfinite(event_end)
                or event_end <= event_start
            ):
                continue

            trial_num = (
                timing_p.iloc[pos]["trial_num"]
                if "trial_num" in timing_p.columns
                else pos + 1
            )

            isi_value = (
                timing_p.iloc[pos][isi_col]
                if isi_col in timing_p.columns
                else np.nan
            )

            # Full trial window: event -> next event
            trial_mask = (
                (timestamps >= event_start)
                & (timestamps < event_end)
            )

            if trial_mask.sum() < min_samples:
                continue

            t_trial = timestamps[trial_mask] - event_start
            y_raw_trial = raw_emg[trial_mask]
            y_processed_trial = processed_emg[trial_mask]

            # Pre-event baseline window
            baseline_start = event_start + baseline_window_s[0]
            baseline_end = event_start + baseline_window_s[1]

            baseline_mask = (
                (timestamps >= baseline_start)
                & (timestamps < baseline_end)
            )

            if baseline_mask.sum() >= min_samples:
                y_raw_baseline = raw_emg[baseline_mask]
                baseline_center = np.nanmedian(y_raw_baseline)
                rms_baseline = _rms(y_raw_baseline - baseline_center)
            else:
                y_raw_baseline = np.array([])
                baseline_center = np.nanmedian(y_raw_trial)
                rms_baseline = np.nan

            # Center raw EMG using pre-event baseline if available.
            y_raw_centered = y_raw_trial - baseline_center

            row = {
                "participant_id": participant_id,
                "base_participant": get_base_participant(participant_id),
                "session_type": get_session_type(participant_id),
                "trial_num": trial_num,
                "trial_position": pos + 1,
                "event_start": event_start,
                "event_end": event_end,
                "trial_duration_s": event_end - event_start,
                "isi": timing_p.iloc[pos]["isi"] if "isi" in timing_p.columns else np.nan,
                "isi_bin": timing_p.iloc[pos]["isi_bin"] if "isi_bin" in timing_p.columns else isi_value,
                "n_samples": int(trial_mask.sum()),
                "fs_hz": fs,

                # Amplitude features
                "rms_response": _rms(y_raw_centered),
                "mav_response": _mav(y_raw_centered),
                "iemg_response": _iemg(y_raw_centered, fs),
                "peak_abs_response": _peak_abs(y_raw_centered),
                "time_to_peak_s": _time_to_peak(y_raw_centered, t_trial),

                # Features based on your processed/rectified EMG
                "mean_processed_response": np.nanmean(y_processed_trial),
                "max_processed_response": np.nanmax(y_processed_trial),

                # Pre-event baseline
                "rms_pre_event_baseline": rms_baseline,
            }

            if np.isfinite(row["rms_pre_event_baseline"]):
                row["rms_response_minus_pre_event"] = (
                    row["rms_response"]
                    - row["rms_pre_event_baseline"]
                )
            else:
                row["rms_response_minus_pre_event"] = np.nan

            if frequency_enabled:
                row.update(
                    _spectral_features(
                        y=y_raw_centered,
                        fs=fs,
                        min_freq_hz=freq_cfg.get("min_freq_hz", 5.0),
                        max_freq_hz=freq_cfg.get("max_freq_hz", None),
                        welch_nperseg=freq_cfg.get("welch_nperseg", 256),
                    )
                )

            rows.append(row)

    features = pd.DataFrame(rows)

    if features.empty:
        return features

    features = features.sort_values(
        ["participant_id", "trial_num"]
    ).reset_index(drop=True)

    baseline_cfg = cfg.get("session_baseline", {})
    n_baseline_trials = baseline_cfg.get("n_trials", 10)
    baseline_stat = baseline_cfg.get("stat", "median")

    base_feature_cols_to_normalize = [
        "rms_response",
        "rms_response_minus_pre_event",
        "mav_response",
        "iemg_response",
        "peak_abs_response",
        "mean_processed_response",
        "max_processed_response",
        "mdf_response_hz",
        "mnf_response_hz",
        "spectral_power_response",
    ]

    base_feature_cols_to_normalize = [
        col for col in base_feature_cols_to_normalize
        if col in features.columns
    ]

    # ------------------------------------------------------------
    # Rolling variability features
    # ------------------------------------------------------------
    variability_cfg = cfg.get("variability", {})

    if variability_cfg.get("enabled", True):
        rolling_window_trials = variability_cfg.get(
            "rolling_window_trials",
            10,
        )

        rolling_min_periods = variability_cfg.get(
            "rolling_min_periods",
            5,
        )

        variability_features = variability_cfg.get(
            "features",
            ["rms_response"],
        )

        variability_features = [
            col for col in variability_features
            if col in features.columns
        ]

        features = add_rolling_feature_variability(
            features=features,
            feature_cols=variability_features,
            window_trials=rolling_window_trials,
            min_periods=rolling_min_periods,
        )

        rolling_feature_cols = []

        for feature_col in variability_features:
            rolling_feature_cols.append(
                f"{feature_col}_rolling_sd_{rolling_window_trials}"
            )
            rolling_feature_cols.append(
                f"{feature_col}_rolling_cv_{rolling_window_trials}_pct"
            )

        rolling_feature_cols = [
            col for col in rolling_feature_cols
            if col in features.columns
        ]

    else:
        rolling_feature_cols = []

    feature_cols_to_normalize = (
        base_feature_cols_to_normalize
        + rolling_feature_cols
    )

    features = add_first_trials_normalization(
        features=features,
        feature_cols=feature_cols_to_normalize,
        n_trials=n_baseline_trials,
        stat=baseline_stat,
    )

    block_size = cfg.get("plot", {}).get("block_size", 40)

    if block_size is not None:
        features["trial_block"] = (
            ((features["trial_num"].astype(float) - 1) // block_size) + 1
        )

    return features