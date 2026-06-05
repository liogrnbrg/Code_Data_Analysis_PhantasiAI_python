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


def integrate_accel_one_trial(
    t,
    accel,
    baseline_value=None,
    force_zero_velocity_end=True,
    detrend_position=True,
):
    """
    Integrate acceleration within one trial.

    accel -> velocity proxy -> position proxy

    These are exploratory proxies, not validated physical velocity/position.
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

        data_p = data[data["participant_id"] == participant_id].sort_values("timestamp")
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

    data["velocity_norm"] = np.sqrt(
        data["velocity_x"] ** 2 + data["velocity_y"] ** 2 + data["velocity_z"] ** 2
    )

    data["position_norm"] = np.sqrt(
        data["position_x"] ** 2 + data["position_y"] ** 2 + data["position_z"] ** 2
    )

    return data

def extract_trial_amplitude_metrics(
    signal_data,
    timing_data,
    signal_vars=(
        "accel_x", "accel_y", "accel_z",
        "velocity_x", "velocity_y", "velocity_z",
        "position_x", "position_y", "position_z",
    ),
):
    """
    Extract per-trial movement amplitude metrics.

    For each trial:
        amplitude = max(signal) - min(signal)

    Trial window:
        current event -> next event

    Dominant-axis metrics:
        The dominant axis is selected once per participant/session,
        based on the largest mean amplitude across all trials.
        This avoids changing the selected axis from trial to trial.
    """

    rows = []

    for participant_id in timing_data["participant_id"].dropna().unique():

        signal_p = signal_data[
            signal_data["participant_id"] == participant_id
        ].sort_values("timestamp").reset_index(drop=True)

        timing_p = timing_data[
            timing_data["participant_id"] == participant_id
        ].sort_values("event").reset_index(drop=True)

        if len(signal_p) == 0 or len(timing_p) < 2:
            continue

        timestamps = signal_p["timestamp"].to_numpy(dtype=float)
        participant_rows = []

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

            row = {
                "participant_id": participant_id,
                "trial_num": timing_p.loc[i, "trial_num"] if "trial_num" in timing_p.columns else i + 1,
                "isi": timing_p.loc[i, "isi"],
                "isi_bin": timing_p.loc[i, "isi_bin"] if "isi_bin" in timing_p.columns else np.nan,
                "event": event_start,
                "next_event": event_end,
                "trial_duration_s": event_end - event_start,
            }

            for signal_var in signal_vars:

                if signal_var not in signal_p.columns:
                    continue

                y = signal_p.loc[trial_mask, signal_var].to_numpy(dtype=float)
                y = y[np.isfinite(y)]

                if len(y) < 3:
                    row[f"{signal_var}_amp"] = np.nan
                    row[f"{signal_var}_abs_peak"] = np.nan
                    continue

                row[f"{signal_var}_amp"] = np.nanmax(y) - np.nanmin(y)
                row[f"{signal_var}_abs_peak"] = np.nanmax(np.abs(y))

            # 3D amplitude across axes
            for prefix in ["accel", "velocity", "position"]:

                x_col = f"{prefix}_x_amp"
                y_col = f"{prefix}_y_amp"
                z_col = f"{prefix}_z_amp"

                if x_col in row and y_col in row and z_col in row:
                    row[f"{prefix}_3d_amp"] = np.sqrt(
                        row[x_col] ** 2
                        + row[y_col] ** 2
                        + row[z_col] ** 2
                    )

            participant_rows.append(row)

        if len(participant_rows) == 0:
            continue

        participant_df = pd.DataFrame(participant_rows)

        # Select one dominant axis per participant/session
        for prefix in ["accel", "velocity", "position"]:

            axis_amp_cols = {
                "x": f"{prefix}_x_amp",
                "y": f"{prefix}_y_amp",
                "z": f"{prefix}_z_amp",
            }

            available_cols = {
                axis: col
                for axis, col in axis_amp_cols.items()
                if col in participant_df.columns
            }

            if len(available_cols) == 0:
                continue

            mean_amp_by_axis = {
                axis: participant_df[col].mean(skipna=True)
                for axis, col in available_cols.items()
            }

            session_dominant_axis = max(mean_amp_by_axis, key=mean_amp_by_axis.get)
            session_dominant_col = available_cols[session_dominant_axis]

            participant_df[f"{prefix}_session_dominant_axis"] = session_dominant_axis
            participant_df[f"{prefix}_session_dominant_amp"] = participant_df[session_dominant_col]

        rows.append(participant_df)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)