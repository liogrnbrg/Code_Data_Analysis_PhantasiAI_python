from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
FUNCTIONS_DIR = PROJECT_DIR / "functions"

sys.path.append(str(FUNCTIONS_DIR))

from utils.config import get_config
from loading.load_data import load_timing_data
from preprocessing.isi_binning import add_isi_bin_column
from plotting.plot_timing_features import (
    plot_peak_amplitude_regression_by_participant,
    plot_peak_amplitude_boxplot_by_isi,
    plot_event_peak_delay_regression_by_participant,
    plot_event_peak_delay_boxplot_by_isi,
    plot_delta_peak_amplitude_vs_delta_isi_by_participant,
    plot_peak_amplitude_vs_recent_isi_context_by_participant,
    plot_current_peak_amplitude_vs_delta_isi_by_participant,
    plot_delta_peak_amplitude_vs_delta_isi_regression_by_participant,
    plot_delta_peak_amplitude_by_delta_isi_sign_by_participant,
    plot_peak_amplitude_regressions_combined,
    plot_event_peak_delay_regressions_combined,
)

C = get_config()

DATA_DIR = Path(C["data"]["path"])
PLOTS_DIR = PROJECT_DIR / "outputs" / "plots" / "timings_analysis"

DATA_DIR = Path(C["data"]["path"]).expanduser()

if not DATA_DIR.exists():
    DATA_DIR = PROJECT_DIR / "data"

if not DATA_DIR.exists():
    raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

def main():
    print("Loading timing data...")

    C = get_config()
    subject_colors = C["plot"]["subject_colors"]

    timing_data = load_timing_data(DATA_DIR)
    timing_data = add_isi_bin_column(timing_data, C)

    print("ISI bins found:")
    print(timing_data.groupby("participant_id")["isi_bin"].value_counts(dropna=False))

    normalizations = ["first", "zscore"]

    print("Plotting peak amplitude regressions...")
    for norm in normalizations:
        plot_peak_amplitude_regression_by_participant(
            timing_data=timing_data,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
            normalization=norm,
        )

    print("Plotting peak amplitude boxplots...")
    for norm in normalizations:
        plot_peak_amplitude_boxplot_by_isi(
            timing_data=timing_data,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
            normalization=norm,
        )

    print("Plotting event-to-peak delay regressions...")
    plot_event_peak_delay_regression_by_participant(
        timing_data=timing_data,
        plots_dir=PLOTS_DIR,
        subject_colors=subject_colors,
    )

    print("Plotting event-to-peak delay boxplots...")
    plot_event_peak_delay_boxplot_by_isi(
        timing_data=timing_data,
        plots_dir=PLOTS_DIR,
        subject_colors=subject_colors,
    )

    print("Plotting delta peak amplitude vs delta ISI regression...")
    plot_delta_peak_amplitude_vs_delta_isi_regression_by_participant(
    timing_data=timing_data,
    plots_dir=PLOTS_DIR,
    subject_colors=subject_colors,
    )

    print("Plotting delta peak amplitude vs delta ISI sign...")
    plot_delta_peak_amplitude_by_delta_isi_sign_by_participant(
        timing_data=timing_data,
        plots_dir=PLOTS_DIR,
        subject_colors=subject_colors,
    )

    print("Plotting delta peak amplitude vs delta ISI...")
    plot_delta_peak_amplitude_vs_delta_isi_by_participant(
        timing_data=timing_data,
        plots_dir=PLOTS_DIR,
        subject_colors=subject_colors,
    )

    print("Plotting peak amplitude vs recent ISI context...")
    for norm in ["zscore", "first"]:
        plot_peak_amplitude_vs_recent_isi_context_by_participant(
            timing_data=timing_data,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
            n_previous_trials=2,
            normalization=norm,
        )

    print("Plotting current peak amplitude vs delta ISI...")
    for norm in ["zscore", "first"]:
        plot_current_peak_amplitude_vs_delta_isi_by_participant(
            timing_data=timing_data,
            plots_dir=PLOTS_DIR,
            subject_colors=subject_colors,
            normalization=norm,
        )

    # ============================================================
    # Combined participant regression plots
    # ============================================================

    plot_event_peak_delay_regressions_combined(
        timing_data=timing_data,
        plots_dir=PLOTS_DIR,
        subject_colors=subject_colors,
        alpha=0.05,
        hac_maxlags=10,
        band="sd",          # default
        sd_multiplier=1.0,  # ±1 SD
    )

    plot_peak_amplitude_regressions_combined(
        timing_data=timing_data,
        plots_dir=PLOTS_DIR,
        subject_colors=subject_colors,
        normalization="first",
        alpha=0.05,
        hac_maxlags=10,
        band="sd",          # default
        sd_multiplier=1.0,
    )

    plot_peak_amplitude_regressions_combined(
        timing_data=timing_data,
        plots_dir=PLOTS_DIR,
        subject_colors=subject_colors,
        normalization="zscore",
        alpha=0.05,
        hac_maxlags=10,
        band="sd",          # default
        sd_multiplier=1.0,
    )

    print("Done. Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()