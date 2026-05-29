import numpy as np


def plot_mean_sd_band(ax, x, profiles, color, face_alpha=0.16, line_width=2.2):
    profiles = np.asarray(profiles, dtype=float)
    x = np.asarray(x, dtype=float)

    if profiles.size == 0:
        return False

    valid_rows = np.any(np.isfinite(profiles), axis=1)
    profiles = profiles[valid_rows, :]

    if profiles.size == 0:
        return False

    valid_cols = np.any(np.isfinite(profiles), axis=0)

    if not np.any(valid_cols):
        return False

    x_plot = x[valid_cols]
    profiles_plot = profiles[:, valid_cols]

    mean_y = np.nanmean(profiles_plot, axis=0)
    sd_y = np.nanstd(profiles_plot, axis=0)

    upper = mean_y + sd_y
    lower = mean_y - sd_y

    ax.fill_between(
        x_plot,
        lower,
        upper,
        color=color,
        alpha=face_alpha,
        linewidth=0,
    )

    ax.plot(
        x_plot,
        mean_y,
        color=color,
        linewidth=line_width,
    )

    return True