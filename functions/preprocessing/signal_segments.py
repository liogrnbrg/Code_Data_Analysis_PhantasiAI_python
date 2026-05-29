from pathlib import Path


def select_signal_segment(df, fraction_to_plot=1 / 20, segment_position="start"):
    """
    Select a fraction of a signal table for quick visual inspection.
    """

    n_samples = len(df)
    n_segment = max(1, int(n_samples * fraction_to_plot))

    if segment_position == "start":
        idx_start = 0
    elif segment_position == "middle":
        idx_start = max(0, (n_samples - n_segment) // 2)
    elif segment_position == "end":
        idx_start = max(0, n_samples - n_segment)
    else:
        raise ValueError(f"Unknown segment_position: {segment_position}")

    idx_end = min(idx_start + n_segment, n_samples)

    return df.iloc[idx_start:idx_end].copy()