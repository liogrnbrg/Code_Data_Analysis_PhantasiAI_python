import numpy as np


def normalize_signal(y, normalization):
    y = np.asarray(y, dtype=float)

    if normalization == "raw":
        return y

    if normalization == "first":
        first_valid = y[np.isfinite(y)][0]

        if first_valid == 0:
            print("Warning: first valid value is zero. Returning raw values.")
            return y

        return y / first_valid

    if normalization == "zscore":
        mean_y = np.nanmean(y)
        std_y = np.nanstd(y)

        if std_y == 0 or not np.isfinite(std_y):
            print("Warning: standard deviation is zero or NaN. Returning raw values.")
            return y

        return (y - mean_y) / std_y

    raise ValueError(f"Unknown normalization: {normalization}")


def get_normalization_label(normalization):
    if normalization == "first":
        return "Peak amplitude normalized to first trial", "peak amplitude / first trial"

    if normalization == "zscore":
        return "Peak amplitude z-scored within participant", "Peak amplitude (z-score)"

    if normalization == "raw":
        return "Raw peak amplitude", "Peak amplitude"

    raise ValueError(f"Unknown normalization: {normalization}")