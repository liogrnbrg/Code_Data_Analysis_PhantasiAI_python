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