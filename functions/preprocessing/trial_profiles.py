import numpy as np


def build_trial_to_next_event_profiles(
    signal_df,
    timing_df,
    signal_var,
    target_isi,
    config,
):
    """
    Build profiles from current event to next event.

    Uses timing_df event[i] as trial start and timing_df event[i+1] as trial end.
    Grouping is done using isi_bin if available.
    """

    profile_cfg = config["emg_patterns"]["profile"]

    n_points = profile_cfg["n_points"]
    time_mode = profile_cfg.get("time_mode", "seconds")
    duration_strategy = profile_cfg.get("duration_strategy", "median_duration")
    max_trials = profile_cfg.get("max_trials_per_isi", None)

    timing_df = timing_df.sort_values("event").reset_index(drop=True)
    signal_df = signal_df.sort_values("timestamp").reset_index(drop=True)

    isi_col = "isi_bin" if "isi_bin" in timing_df.columns else "isi"

    # positions, not index labels
    trial_positions = np.where(timing_df[isi_col].to_numpy(dtype=float) == float(target_isi))[0]

    # Exclude last row because it has no next event
    trial_positions = trial_positions[trial_positions < len(timing_df) - 1]

    if max_trials is not None and len(trial_positions) > max_trials:
        trial_positions = trial_positions[:max_trials]

    if len(trial_positions) == 0:
        return np.array([]), np.empty((0, n_points)), np.array([])

    durations = []

    for pos in trial_positions:
        event_start = timing_df.iloc[pos]["event"]
        event_end = timing_df.iloc[pos + 1]["event"]

        if np.isfinite(event_start) and np.isfinite(event_end) and event_end > event_start:
            durations.append(event_end - event_start)

    durations = np.asarray(durations, dtype=float)

    if durations.size == 0:
        return np.array([]), np.empty((0, n_points)), np.array([])

    if time_mode == "normalized":
        t_grid = np.linspace(0, 1, n_points)

    elif time_mode == "seconds":
        if duration_strategy == "median_duration":
            t_end = np.nanmedian(durations)
        elif duration_strategy == "max_duration":
            t_end = np.nanmax(durations)
        elif duration_strategy == "target_isi":
            t_end = float(target_isi)
        else:
            raise ValueError(f"Unknown duration_strategy: {duration_strategy}")

        t_grid = np.linspace(0, t_end, n_points)

    else:
        raise ValueError(f"Unknown time_mode: {time_mode}")

    profiles = []
    trial_nums = []

    timestamps = signal_df["timestamp"].to_numpy(dtype=float)
    signal = signal_df[signal_var].to_numpy(dtype=float)

    for pos in trial_positions:
        event_start = timing_df.iloc[pos]["event"]
        event_end = timing_df.iloc[pos + 1]["event"]

        if not np.isfinite(event_start) or not np.isfinite(event_end) or event_end <= event_start:
            continue

        idx_window = (timestamps >= event_start) & (timestamps < event_end)

        if idx_window.sum() < 5:
            continue

        t_rel = timestamps[idx_window] - event_start
        y_raw = signal[idx_window]

        if time_mode == "normalized":
            duration = event_end - event_start
            x_trial = t_rel / duration
        else:
            x_trial = t_rel

        valid = np.isfinite(x_trial) & np.isfinite(y_raw)

        if valid.sum() < 5:
            continue

        x_trial = x_trial[valid]
        y_raw = y_raw[valid]

        order = np.argsort(x_trial)
        x_trial = x_trial[order]
        y_raw = y_raw[order]

        x_unique, unique_idx = np.unique(x_trial, return_index=True)
        y_unique = y_raw[unique_idx]

        if len(x_unique) < 5:
            continue

        y_interp = np.interp(t_grid, x_unique, y_unique, left=np.nan, right=np.nan)

        profiles.append(y_interp)

        if "trial_num" in timing_df.columns:
            trial_nums.append(timing_df.iloc[pos]["trial_num"])
        else:
            trial_nums.append(pos)

    if len(profiles) == 0:
        return t_grid, np.empty((0, n_points)), np.array([])

    return t_grid, np.vstack(profiles), np.asarray(trial_nums)