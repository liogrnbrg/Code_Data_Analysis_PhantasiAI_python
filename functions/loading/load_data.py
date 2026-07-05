# load_data.py

from pathlib import Path

import pandas as pd


def load_data(data_dir):
    """
    Load all timing and signal files found recursively in data_dir.
    """

    data_dir = Path(data_dir)

    print(f"Loading data from: {data_dir.resolve()}")

    signal_data = load_emg_accel_data(data_dir)
    timing_data = load_timing_data(data_dir)

    signal_participants = sorted(
        signal_data["participant_id"].dropna().unique()
    )

    timing_participants = sorted(
        timing_data["participant_id"].dropna().unique()
    )

    print("\nParticipants in signal data:")
    for participant_id in signal_participants:
        print(f"  - {participant_id}")

    print("\nParticipants in timing data:")
    for participant_id in timing_participants:
        print(f"  - {participant_id}")

    signal_only = sorted(
        set(signal_participants) - set(timing_participants)
    )

    timing_only = sorted(
        set(timing_participants) - set(signal_participants)
    )

    if signal_only:
        print("\nWarning — signal data without timing data:")
        for participant_id in signal_only:
            print(f"  - {participant_id}")

    if timing_only:
        print("\nWarning — timing data without signal data:")
        for participant_id in timing_only:
            print(f"  - {participant_id}")

    return signal_data, timing_data

def load_timing_data(data_dir):
    """
    Load every timings.csv file found recursively in data_dir.
    """

    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("**/timings.csv"))

    if not files:
        raise FileNotFoundError(
            f"No timings.csv files found in {data_dir}"
        )

    rows = []

    for file in files:
        participant_id = file.parent.name

        df = pd.read_csv(file, sep=None, engine="python")

        print(f"\nLoaded participant: {participant_id}")
        print(f"Exact file: {file.resolve()}")
        print("Raw columns:", [repr(c) for c in df.columns])

        df.columns = (
            df.columns
            .astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        # Detect columns that became duplicated after cleaning.
        duplicated_columns = df.columns[df.columns.duplicated()].tolist()

        if duplicated_columns:
            raise ValueError(
                f"Duplicated columns after cleaning in {file}: "
                f"{duplicated_columns}"
            )

        print("Clean columns:", df.columns.tolist())

        if "event" in df.columns:
            print(
                "event:",
                df["event"].notna().sum(),
                "non-NaN | examples:",
                df["event"].dropna().head(3).tolist(),
            )
        else:
            print('WARNING: no column named exactly "event"')

        if "isi" in df.columns:
            print(
                "isi:",
                df["isi"].notna().sum(),
                "non-NaN | values:",
                sorted(df["isi"].dropna().unique())[:10],
            )
        else:
            print('WARNING: no column named exactly "isi"')

        if "participant_id" in df.columns:
            df["participant_id"] = participant_id
        else:
            df.insert(0, "participant_id", participant_id)

        rows.append(df)

    out = pd.concat(rows, ignore_index=True, sort=False)

    # Explicit numeric conversion.
    for column in ["trial_num", "event", "isi"]:
        if column in out.columns:
            out[column] = pd.to_numeric(
                out[column],
                errors="coerce",
            )

    print("\nAfter concatenation:")

    print(
        out.groupby("participant_id").agg(
            n_rows=("participant_id", "size"),
            n_event=("event", lambda x: x.notna().sum()),
            n_isi=("isi", lambda x: x.notna().sum()),
        )
    )

    out = (
        out
        .sort_values(["participant_id", "event"])
        .reset_index(drop=True)
    )

    return out


def load_emg_accel_data(data_dir):
    """
    Load every emg_accel.csv file found recursively in data_dir.
    """

    data_dir = Path(data_dir)

    files = sorted(data_dir.glob("**/emg_accel.csv"))

    if not files:
        raise FileNotFoundError(
            f"No emg_accel.csv files found in {data_dir}"
        )

    rows = []

    for file in files:
        participant_id = file.parent.name

        df = pd.read_csv(file)

        # Avoid duplicate participant_id columns if the CSV already has one.
        if "participant_id" in df.columns:
            df["participant_id"] = participant_id
        else:
            df.insert(0, "participant_id", participant_id)

        rows.append(df)

    out = pd.concat(rows, ignore_index=True)

    out = (
        out
        .sort_values(["participant_id", "timestamp"])
        .reset_index(drop=True)
    )

    return out