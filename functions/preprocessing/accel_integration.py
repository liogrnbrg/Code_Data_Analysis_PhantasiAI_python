import numpy as np
import pandas as pd


def cumulative_trapezoid_np(y, t):
    """
    Simple cumulative trapezoidal integration without scipy.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out


def add_velocity_and_position_proxies(
    data,
    accel_cols=("accel_x", "accel_y", "accel_z"),
    baseline_mode="median",
    detrend_velocity=True,
    detrend_position=True,
):
    """
    Add velocity and position proxy columns by integrating acceleration.

    Steps:
        1. remove acceleration baseline per axis
        2. integrate acceleration -> velocity proxy
        3. optionally remove linear velocity drift
        4. integrate velocity -> position proxy
        5. optionally remove linear position drift

    Important:
        These are exploratory proxies, not true physical velocity/position.
    """

    data = data.sort_values("timestamp").copy()

    t = data["timestamp"].to_numpy(dtype=float)
    t = t - t[0]

    for accel_col in accel_cols:
        axis = accel_col.replace("accel_", "")

        a = data[accel_col].to_numpy(dtype=float)

        if baseline_mode == "median":
            baseline = np.nanmedian(a)
        elif baseline_mode == "mean":
            baseline = np.nanmean(a)
        else:
            baseline = 0.0

        a_centered = a - baseline

        velocity = cumulative_trapezoid_np(a_centered, t)

        if detrend_velocity:
            velocity_drift = np.linspace(velocity[0], velocity[-1], len(velocity))
            velocity = velocity - velocity_drift

        position = cumulative_trapezoid_np(velocity, t)

        if detrend_position:
            position_drift = np.linspace(position[0], position[-1], len(position))
            position = position - position_drift

        data[f"velocity_{axis}"] = velocity
        data[f"position_{axis}"] = position

    return data

import numpy as np


def cumulative_trapezoid_np(y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out


def integrate_accel_one_trial(
    t,
    accel,
    baseline_value=None,
    force_zero_velocity_end=True,
    detrend_position=True,
):
    """
    Trial-wise exploratory integration.

    accel -> velocity proxy -> position proxy

    Important:
    These are proxies, not validated physical velocity/position.
    """

    t = np.asarray(t, dtype=float)
    accel = np.asarray(accel, dtype=float)

    valid = np.isfinite(t) & np.isfinite(accel)
    t = t[valid]
    accel = accel[valid]

    if len(t) < 3:
        return None, None, None

    t = t - t[0]

    if baseline_value is None:
        baseline_value = np.nanmedian(accel)

    accel_centered = accel - baseline_value

    velocity = cumulative_trapezoid_np(accel_centered, t)

    if force_zero_velocity_end:
        velocity_drift = np.linspace(velocity[0], velocity[-1], len(velocity))
        velocity = velocity - velocity_drift

    position = cumulative_trapezoid_np(velocity, t)

    if detrend_position:
        position_drift = np.linspace(position[0], position[-1], len(position))
        position = position - position_drift

    return t, velocity, position


def add_trialwise_velocity_position_proxies(
    data,
    timing_data,
    accel_cols=("accel_x", "accel_y", "accel_z"),
    baseline_window_s=(-0.5, 0),
    force_zero_velocity_end=True,
    detrend_position=True,
):
    """
    Add trial-wise velocity and position proxy columns.

    For each trial:
        event -> next event
        baseline = mean acceleration before event
        acceleration is integrated only inside that trial window
    """

    data = data.sort_values("timestamp").copy()

    for accel_col in accel_cols:
        axis = accel_col.replace("accel_", "")
        data[f"velocity_{axis}"] = np.nan
        data[f"position_{axis}"] = np.nan

    participants = data["participant_id"].dropna().unique()

    for participant_id in participants:

        idx_p = data["participant_id"] == participant_id

        data_p = data.loc[idx_p].sort_values("timestamp")
        timing_p = timing_data[
            timing_data["participant_id"] == participant_id
        ].sort_values("event").reset_index(drop=True)

        if len(timing_p) < 2:
            continue

        timestamps = data_p["timestamp"].to_numpy(dtype=float)
        data_p_index = data_p.index.to_numpy()

        for i in range(len(timing_p) - 1):

            event_start = timing_p.loc[i, "event"]
            event_end = timing_p.loc[i + 1, "event"]

            if not np.isfinite(event_start) or not np.isfinite(event_end):
                continue

            if event_end <= event_start:
                continue

            trial_mask = (timestamps >= event_start) & (timestamps < event_end)

            if trial_mask.sum() < 3:
                continue

            trial_indices = data_p_index[trial_mask]
            t_trial = timestamps[trial_mask]

            baseline_start = event_start + baseline_window_s[0]
            baseline_end = event_start + baseline_window_s[1]
            baseline_mask = (timestamps >= baseline_start) & (timestamps < baseline_end)

            for accel_col in accel_cols:

                axis = accel_col.replace("accel_", "")
                accel_all = data_p[accel_col].to_numpy(dtype=float)

                if baseline_mask.sum() >= 3:
                    baseline_value = np.nanmean(accel_all[baseline_mask])
                else:
                    baseline_value = np.nanmedian(accel_all[trial_mask])

                accel_trial = accel_all[trial_mask]

                _, velocity, position = integrate_accel_one_trial(
                    t=t_trial,
                    accel=accel_trial,
                    baseline_value=baseline_value,
                    force_zero_velocity_end=force_zero_velocity_end,
                    detrend_position=detrend_position,
                )

                if velocity is None:
                    continue

                data.loc[trial_indices, f"velocity_{axis}"] = velocity
                data.loc[trial_indices, f"position_{axis}"] = position

    return data