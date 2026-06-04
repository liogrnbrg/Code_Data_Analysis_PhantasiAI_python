import numpy as np
import pandas as pd


def add_accel_norm(accel_data):
    """
    Add acceleration norm to the dataframe.

    accel_norm = sqrt(accel_x^2 + accel_y^2 + accel_z^2)
    """

    accel_data = accel_data.copy()

    accel_data["accel_norm"] = np.sqrt(
        accel_data["accel_x"] ** 2
        + accel_data["accel_y"] ** 2
        + accel_data["accel_z"] ** 2
    )

    return accel_data


def add_accel_tilt_angles(accel_data):
    """
    Add simple accelerometer-based tilt angles.

    These are proxy angles based on the accelerometer vector.
    They should be interpreted as sensor/segment inclination proxies,
    not as true joint angles.
    """

    accel_data = accel_data.copy()

    ax = accel_data["accel_x"].to_numpy(dtype=float)
    ay = accel_data["accel_y"].to_numpy(dtype=float)
    az = accel_data["accel_z"].to_numpy(dtype=float)

    accel_data["tilt_x_deg"] = np.degrees(np.arctan2(ax, np.sqrt(ay**2 + az**2)))
    accel_data["tilt_y_deg"] = np.degrees(np.arctan2(ay, np.sqrt(ax**2 + az**2)))
    accel_data["tilt_z_deg"] = np.degrees(np.arctan2(np.sqrt(ax**2 + ay**2), az))

    return accel_data


def extract_accel_behavior_metrics(
    accel_data,
    timing_data,
    signal_var="accel_norm",
    baseline_window_s=(-0.5, 0),
    response_window_s=(0, None),
):
    """
    Extract behavioral movement metrics from accelerometer data for each trial.

    For each trial:
        start = event time
        end   = next event time

    Metrics:
        baseline_mean
        peak_value
        peak_relative_to_baseline
        peak_latency_s
        auc_relative_to_baseline
        signal_range
    """

    rows = []

    for participant_id in timing_data["participant_id"].dropna().unique():

        accel_p = accel_data[
            accel_data["participant_id"] == participant_id
        ].sort_values("timestamp").reset_index(drop=True)

        timing_p = timing_data[
            timing_data["participant_id"] == participant_id
        ].sort_values("event").reset_index(drop=True)

        if len(accel_p) == 0 or len(timing_p) < 2:
            continue

        timestamps = accel_p["timestamp"].to_numpy(dtype=float)
        signal = accel_p[signal_var].to_numpy(dtype=float)

        for i in range(len(timing_p) - 1):

            event_start = timing_p.loc[i, "event"]
            event_end = timing_p.loc[i + 1, "event"]

            if not np.isfinite(event_start) or not np.isfinite(event_end):
                continue

            if event_end <= event_start:
                continue

            baseline_start = event_start + baseline_window_s[0]
            baseline_end = event_start + baseline_window_s[1]

            response_start = event_start + response_window_s[0]

            if response_window_s[1] is None:
                response_end = event_end
            else:
                response_end = event_start + response_window_s[1]

            baseline_idx = (timestamps >= baseline_start) & (timestamps < baseline_end)
            response_idx = (timestamps >= response_start) & (timestamps < response_end)

            baseline_signal = signal[baseline_idx]
            response_signal = signal[response_idx]
            response_time = timestamps[response_idx] - event_start

            if len(response_signal) < 3:
                continue

            baseline_mean = np.nanmean(baseline_signal) if len(baseline_signal) > 0 else np.nan

            peak_idx = np.nanargmax(response_signal)
            peak_value = response_signal[peak_idx]
            peak_latency_s = response_time[peak_idx]

            if np.isfinite(baseline_mean):
                response_centered = response_signal - baseline_mean
                peak_relative = peak_value - baseline_mean
                auc_relative = np.trapezoid(response_centered, response_time)
            else:
                peak_relative = np.nan
                auc_relative = np.nan

            signal_range = np.nanmax(response_signal) - np.nanmin(response_signal)

            row = {
                "participant_id": participant_id,
                "trial_num": timing_p.loc[i, "trial_num"] if "trial_num" in timing_p.columns else i + 1,
                "isi": timing_p.loc[i, "isi"],
                "isi_bin": timing_p.loc[i, "isi_bin"] if "isi_bin" in timing_p.columns else np.nan,
                "event": event_start,
                "signal_var": signal_var,
                "baseline_mean": baseline_mean,
                "accel_peak": peak_value,
                "accel_peak_relative": peak_relative,
                "accel_peak_latency_s": peak_latency_s,
                "accel_auc_relative": auc_relative,
                "accel_range": signal_range,
            }

            rows.append(row)

    return pd.DataFrame(rows)


def extract_angle_rom_metrics(
    accel_data,
    timing_data,
    angle_vars=("tilt_x_deg", "tilt_y_deg", "tilt_z_deg"),
):
    """
    Extract ROM proxy metrics from accelerometer-derived tilt angles.

    ROM proxy = max(angle) - min(angle) within each trial window.

    Important:
    This is not a true joint ROM unless sensor placement/orientation
    and segment calibration justify it.
    """

    all_metrics = []

    for angle_var in angle_vars:

        metrics = extract_accel_behavior_metrics(
            accel_data=accel_data,
            timing_data=timing_data,
            signal_var=angle_var,
            baseline_window_s=(-0.5, 0),
            response_window_s=(0, None),
        )

        if metrics.empty:
            continue

        metrics = metrics.rename(
            columns={
                "accel_peak": f"{angle_var}_peak",
                "accel_peak_relative": f"{angle_var}_peak_relative",
                "accel_peak_latency_s": f"{angle_var}_peak_latency_s",
                "accel_auc_relative": f"{angle_var}_auc_relative",
                "accel_range": f"{angle_var}_rom_proxy",
            }
        )

        keep_cols = [
            "participant_id",
            "trial_num",
            "isi",
            "isi_bin",
            "event",
            f"{angle_var}_peak",
            f"{angle_var}_peak_relative",
            f"{angle_var}_peak_latency_s",
            f"{angle_var}_auc_relative",
            f"{angle_var}_rom_proxy",
        ]

        all_metrics.append(metrics[keep_cols])

    if len(all_metrics) == 0:
        return pd.DataFrame()

    rom_metrics = all_metrics[0]

    for metrics in all_metrics[1:]:
        rom_metrics = rom_metrics.merge(
            metrics,
            on=["participant_id", "trial_num", "isi", "isi_bin", "event"],
            how="outer",
        )

    return rom_metrics

def extract_baseline_corrected_movement_metrics(
    accel_data,
    timing_data,
    baseline_window_s=(-0.5, 0),
    response_window_s=(0, None),
):
    """
    Extract behavioral movement metrics from 3D acceleration.

    For each trial:
        baseline vector = mean accel_x/y/z before event
        movement magnitude = norm(accel(t) - baseline_vector)

    This gives a baseline-corrected movement amplitude proxy.
    """

    rows = []

    for participant_id in timing_data["participant_id"].dropna().unique():

        accel_p = accel_data[
            accel_data["participant_id"] == participant_id
        ].sort_values("timestamp").reset_index(drop=True)

        timing_p = timing_data[
            timing_data["participant_id"] == participant_id
        ].sort_values("event").reset_index(drop=True)

        if len(accel_p) == 0 or len(timing_p) < 2:
            continue

        timestamps = accel_p["timestamp"].to_numpy(dtype=float)

        ax = accel_p["accel_x"].to_numpy(dtype=float)
        ay = accel_p["accel_y"].to_numpy(dtype=float)
        az = accel_p["accel_z"].to_numpy(dtype=float)

        for i in range(len(timing_p) - 1):

            event_start = timing_p.loc[i, "event"]
            event_end = timing_p.loc[i + 1, "event"]

            if not np.isfinite(event_start) or not np.isfinite(event_end):
                continue

            if event_end <= event_start:
                continue

            baseline_start = event_start + baseline_window_s[0]
            baseline_end = event_start + baseline_window_s[1]

            response_start = event_start + response_window_s[0]

            if response_window_s[1] is None:
                response_end = event_end
            else:
                response_end = event_start + response_window_s[1]

            baseline_idx = (timestamps >= baseline_start) & (timestamps < baseline_end)
            response_idx = (timestamps >= response_start) & (timestamps < response_end)

            if baseline_idx.sum() < 3 or response_idx.sum() < 3:
                continue

            baseline_x = np.nanmean(ax[baseline_idx])
            baseline_y = np.nanmean(ay[baseline_idx])
            baseline_z = np.nanmean(az[baseline_idx])

            response_time = timestamps[response_idx] - event_start

            dx = ax[response_idx] - baseline_x
            dy = ay[response_idx] - baseline_y
            dz = az[response_idx] - baseline_z

            movement_mag = np.sqrt(dx**2 + dy**2 + dz**2)

            if len(movement_mag) < 3 or not np.any(np.isfinite(movement_mag)):
                continue

            peak_idx = np.nanargmax(movement_mag)

            movement_peak = movement_mag[peak_idx]
            movement_peak_latency_s = response_time[peak_idx]
            movement_auc = np.trapezoid(movement_mag, response_time)
            movement_mean = np.nanmean(movement_mag)
            movement_sd = np.nanstd(movement_mag)

            rows.append({
                "participant_id": participant_id,
                "trial_num": timing_p.loc[i, "trial_num"] if "trial_num" in timing_p.columns else i + 1,
                "isi": timing_p.loc[i, "isi"],
                "isi_bin": timing_p.loc[i, "isi_bin"] if "isi_bin" in timing_p.columns else np.nan,
                "event": event_start,

                "movement_peak": movement_peak,
                "movement_peak_latency_s": movement_peak_latency_s,
                "movement_auc": movement_auc,
                "movement_mean": movement_mean,
                "movement_sd": movement_sd,

                "baseline_accel_x": baseline_x,
                "baseline_accel_y": baseline_y,
                "baseline_accel_z": baseline_z,
            })

    return pd.DataFrame(rows)

def extract_integrated_accel_displacement_metrics(
    accel_data,
    timing_data,
    accel_axis="accel_z",
    baseline_window_s=(-0.5, 0),
    response_window_s=(0, None),
    detrend_position=True,
):
    """
    Exploratory displacement proxy by double integration of acceleration.

    For each trial:
        1. subtract pre-event acceleration baseline
        2. integrate acceleration -> velocity
        3. integrate velocity -> displacement
        4. extract displacement range / peak

    Important:
        This is NOT true position unless gravity, orientation and drift are corrected.
        It should be interpreted as an exploratory displacement proxy.
    """

    rows = []

    for participant_id in timing_data["participant_id"].dropna().unique():

        accel_p = accel_data[
            accel_data["participant_id"] == participant_id
        ].sort_values("timestamp").reset_index(drop=True)

        timing_p = timing_data[
            timing_data["participant_id"] == participant_id
        ].sort_values("event").reset_index(drop=True)

        if len(accel_p) == 0 or len(timing_p) < 2:
            continue

        timestamps = accel_p["timestamp"].to_numpy(dtype=float)
        accel = accel_p[accel_axis].to_numpy(dtype=float)

        for i in range(len(timing_p) - 1):

            event_start = timing_p.loc[i, "event"]
            event_end = timing_p.loc[i + 1, "event"]

            if not np.isfinite(event_start) or not np.isfinite(event_end):
                continue

            if event_end <= event_start:
                continue

            baseline_start = event_start + baseline_window_s[0]
            baseline_end = event_start + baseline_window_s[1]

            response_start = event_start + response_window_s[0]

            if response_window_s[1] is None:
                response_end = event_end
            else:
                response_end = event_start + response_window_s[1]

            baseline_idx = (timestamps >= baseline_start) & (timestamps < baseline_end)
            response_idx = (timestamps >= response_start) & (timestamps < response_end)

            if baseline_idx.sum() < 3 or response_idx.sum() < 3:
                continue

            t = timestamps[response_idx] - event_start
            a = accel[response_idx]

            baseline_accel = np.nanmean(accel[baseline_idx])
            a = a - baseline_accel

            valid = np.isfinite(t) & np.isfinite(a)
            t = t[valid]
            a = a[valid]

            if len(t) < 3:
                continue

            # Ensure time starts at 0
            t = t - t[0]

            # Integrate acceleration -> velocity
            velocity = np.zeros_like(a)
            velocity[1:] = np.cumsum(
                0.5 * (a[1:] + a[:-1]) * np.diff(t)
            )

            # Remove simple linear drift in velocity:
            # assume velocity should be ~0 at the end of the short trial window
            velocity_drift = np.linspace(velocity[0], velocity[-1], len(velocity))
            velocity_corrected = velocity - velocity_drift

            # Integrate velocity -> displacement
            displacement = np.zeros_like(velocity_corrected)
            displacement[1:] = np.cumsum(
                0.5 * (velocity_corrected[1:] + velocity_corrected[:-1]) * np.diff(t)
            )

            if detrend_position:
                position_drift = np.linspace(displacement[0], displacement[-1], len(displacement))
                displacement = displacement - position_drift

            disp_range = np.nanmax(displacement) - np.nanmin(displacement)
            disp_peak_abs = np.nanmax(np.abs(displacement))
            disp_final = displacement[-1]

            peak_idx = np.nanargmax(np.abs(displacement))
            disp_peak_latency_s = t[peak_idx]

            rows.append({
                "participant_id": participant_id,
                "trial_num": timing_p.loc[i, "trial_num"] if "trial_num" in timing_p.columns else i + 1,
                "isi": timing_p.loc[i, "isi"],
                "isi_bin": timing_p.loc[i, "isi_bin"] if "isi_bin" in timing_p.columns else np.nan,
                "event": event_start,
                "accel_axis": accel_axis,

                "baseline_accel": baseline_accel,
                "disp_range_proxy": disp_range,
                "disp_peak_abs_proxy": disp_peak_abs,
                "disp_final_proxy": disp_final,
                "disp_peak_latency_s": disp_peak_latency_s,
            })

    return pd.DataFrame(rows)