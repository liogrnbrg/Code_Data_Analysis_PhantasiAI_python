from pathlib import Path
import pandas as pd

def load_data(data_dir):
    signal_data = load_emg_accel_data(data_dir)
    timing_data = load_timing_data(data_dir)
    return signal_data, timing_data

def load_timing_data(data_dir):
    data_dir = Path(data_dir)
    files = list(data_dir.glob("**/timings.csv"))

    if not files:
        raise FileNotFoundError(f"No timings.csv files found in {data_dir}")

    rows = []

    for file in files:
        participant_id = file.parent.name
        df = pd.read_csv(file)
        df.insert(0, "participant_id", participant_id)
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["participant_id", "event"]).reset_index(drop=True)

    return out


def load_emg_accel_data(data_dir):
    data_dir = Path(data_dir)
    files = list(data_dir.glob("**/emg_accel.csv"))

    if not files:
        raise FileNotFoundError(f"No emg_accel.csv files found in {data_dir}")

    rows = []

    for file in files:
        participant_id = file.parent.name
        df = pd.read_csv(file)
        df.insert(0, "participant_id", participant_id)
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["participant_id", "timestamp"]).reset_index(drop=True)

    return out