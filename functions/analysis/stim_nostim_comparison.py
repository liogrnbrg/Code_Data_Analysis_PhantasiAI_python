from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress, ttest_rel, wilcoxon
import statsmodels.formula.api as smf


def _split_participant_and_condition(
    participant_id: str,
) -> tuple[str | None, str | None]:
    """
    Convert session identifiers such as:

        Lio_STIM   -> ("Lio", "STIM")
        Lio_NOSTIM -> ("Lio", "NOSTIM")

    Sessions without a STIM/NOSTIM suffix are ignored.
    """
    participant_id = str(participant_id)

    if participant_id.endswith("_NOSTIM"):
        return participant_id.removesuffix("_NOSTIM"), "NOSTIM"

    if participant_id.endswith("_STIM"):
        return participant_id.removesuffix("_STIM"), "STIM"

    return None, None


def prepare_rt_stim_nostim_comparison(
    rt_data: pd.DataFrame,
    rt_col: str = "reaction_time_ms",
    n_baseline_trials: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare long and paired STIM/NOSTIM reaction-time datasets.

    Normalization:
        centered RT =
            RT - mean of the first n valid RTs
            within each participant/session/ISI

    Pairing:
        base participant + global trial_num + isi_bin

    Returns
    -------
    long_data
        One row per original trial, with condition and centered RT.

    paired_data
        One row per matched STIM/NOSTIM trial, with differences.
    """

    required_cols = {
        "participant_id",
        "trial_num",
        "isi_bin",
        "reaction_time_valid",
        rt_col,
    }

    missing = required_cols.difference(rt_data.columns)

    if missing:
        raise KeyError(
            f"Missing required columns in rt_data: {sorted(missing)}"
        )

    long_data = rt_data.copy()

    parsed = long_data["participant_id"].apply(
        _split_participant_and_condition
    )

    long_data["base_participant"] = parsed.str[0]
    long_data["condition"] = parsed.str[1]

    long_data = long_data[
        long_data["condition"].isin(["STIM", "NOSTIM"])
    ].copy()

    if long_data.empty:
        raise ValueError(
            "No matching XX_STIM / XX_NOSTIM sessions were found."
        )

    long_data["rt_valid_value"] = long_data[rt_col].where(
        long_data["reaction_time_valid"].astype(bool)
        & np.isfinite(long_data[rt_col])
    )

    group_cols = [
        "base_participant",
        "condition",
        "isi_bin",
    ]

    baseline_rows = (
        long_data[long_data["rt_valid_value"].notna()]
        .sort_values(group_cols + ["trial_num"])
        .groupby(group_cols, group_keys=False)
        .head(n_baseline_trials)
    )

    baselines = (
        baseline_rows
        .groupby(group_cols)["rt_valid_value"]
        .mean()
        .rename("rt_baseline_ms")
        .reset_index()
    )

    baseline_counts = (
        baseline_rows
        .groupby(group_cols)["rt_valid_value"]
        .size()
        .rename("n_baseline_valid")
        .reset_index()
    )

    baselines = baselines.merge(
        baseline_counts,
        on=group_cols,
        how="left",
    )

    long_data = long_data.merge(
        baselines,
        on=group_cols,
        how="left",
    )

    long_data["reaction_time_centered_ms"] = (
        long_data["rt_valid_value"]
        - long_data["rt_baseline_ms"]
    )

    paired = long_data.pivot_table(
        index=[
            "base_participant",
            "trial_num",
            "isi_bin",
        ],
        columns="condition",
        values=[
            rt_col,
            "reaction_time_centered_ms",
            "reaction_time_valid",
        ],
        aggfunc="first",
    )

    paired.columns = [
        f"{variable}_{condition.lower()}"
        for variable, condition in paired.columns
    ]

    paired = paired.reset_index()

    required_paired_cols = [
        f"{rt_col}_stim",
        f"{rt_col}_nostim",
        "reaction_time_centered_ms_stim",
        "reaction_time_centered_ms_nostim",
        "reaction_time_valid_stim",
        "reaction_time_valid_nostim",
    ]

    for col in required_paired_cols:
        if col not in paired.columns:
            paired[col] = np.nan

    paired["pair_valid"] = (
        paired["reaction_time_valid_stim"].fillna(False).astype(bool)
        & paired["reaction_time_valid_nostim"].fillna(False).astype(bool)
        & np.isfinite(paired[f"{rt_col}_stim"])
        & np.isfinite(paired[f"{rt_col}_nostim"])
    )

    paired["rt_difference_raw_ms"] = (
        paired[f"{rt_col}_stim"]
        - paired[f"{rt_col}_nostim"]
    )

    paired["rt_difference_centered_ms"] = (
        paired["reaction_time_centered_ms_stim"]
        - paired["reaction_time_centered_ms_nostim"]
    )

    # Five global experimental blocks:
    # trials 1–80, 81–160, ..., 321–400
    paired["global_block"] = (
        (paired["trial_num"].astype(int) - 1) // 80
    ) + 1

    long_data["global_block"] = (
        (long_data["trial_num"].astype(int) - 1) // 80
    ) + 1

    return long_data, paired


def add_rolling_rt_variability(
    long_data: pd.DataFrame,
    rt_col: str = "reaction_time_ms",
    rolling_window: int = 10,
    min_periods: int = 5,
) -> pd.DataFrame:
    """
    Compute rolling RT SD separately within each:

        participant × condition × ISI

    The window contains observations from the same ISI, ordered according
    to their global trial number.
    """

    data = long_data.copy()
    output_col = "reaction_time_rolling_sd_ms"
    data[output_col] = np.nan

    group_cols = [
        "base_participant",
        "condition",
        "isi_bin",
    ]

    for _, group_index in data.groupby(group_cols).groups.items():

        group = data.loc[group_index].sort_values("trial_num")

        valid_rt = group[rt_col].where(
            group["reaction_time_valid"].astype(bool)
            & np.isfinite(group[rt_col])
        )

        rolling_sd = valid_rt.rolling(
            window=rolling_window,
            min_periods=min_periods,
            center=True,
        ).std()

        data.loc[group.index, output_col] = rolling_sd

    return data


def prepare_variability_pairs(
    long_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Put STIM and NOSTIM rolling variability values side by side.
    """

    variability_col = "reaction_time_rolling_sd_ms"

    if variability_col not in long_data.columns:
        raise KeyError(
            f"{variability_col} not found. "
            "Run add_rolling_rt_variability first."
        )

    paired = long_data.pivot_table(
        index=[
            "base_participant",
            "trial_num",
            "isi_bin",
            "global_block",
        ],
        columns="condition",
        values=variability_col,
        aggfunc="first",
    )

    paired.columns = [
        f"{variability_col}_{str(condition).lower()}"
        for condition in paired.columns
    ]

    paired = paired.reset_index()

    stim_col = f"{variability_col}_stim"
    nostim_col = f"{variability_col}_nostim"

    for col in [stim_col, nostim_col]:
        if col not in paired.columns:
            paired[col] = np.nan

    paired["variability_pair_valid"] = (
        np.isfinite(paired[stim_col])
        & np.isfinite(paired[nostim_col])
    )

    paired["rt_variability_difference_ms"] = (
        paired[stim_col] - paired[nostim_col]
    )

    return paired


def _paired_effect_size_dz(differences: np.ndarray) -> float:
    """
    Cohen's dz for paired observations.
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]

    if len(differences) < 2:
        return np.nan

    sd_difference = np.std(differences, ddof=1)

    if not np.isfinite(sd_difference) or sd_difference == 0:
        return np.nan

    return np.mean(differences) / sd_difference


def compute_rt_paired_statistics(
    paired_data: pd.DataFrame,
    difference_col: str = "rt_difference_centered_ms",
) -> pd.DataFrame:
    """
    Compute paired STIM/NOSTIM statistics separately for each
    participant and ISI.

    Important:
        These are within-participant, trial-level exploratory statistics.
        They do not represent a population-level inference.
    """

    rows = []

    group_cols = [
        "base_participant",
        "isi_bin",
    ]

    for group_values, group in paired_data.groupby(group_cols):

        base_participant, isi_bin = group_values

        valid = (
            group["pair_valid"].astype(bool)
            & np.isfinite(group[difference_col])
        )

        data = group.loc[valid].sort_values("trial_num")

        differences = data[difference_col].to_numpy(dtype=float)

        stim_col = (
            "reaction_time_centered_ms_stim"
            if difference_col == "rt_difference_centered_ms"
            else "reaction_time_ms_stim"
        )

        nostim_col = (
            "reaction_time_centered_ms_nostim"
            if difference_col == "rt_difference_centered_ms"
            else "reaction_time_ms_nostim"
        )

        stim_values = data[stim_col].to_numpy(dtype=float)
        nostim_values = data[nostim_col].to_numpy(dtype=float)

        result = {
            "base_participant": base_participant,
            "isi_bin": isi_bin,
            "n_valid_pairs": len(data),
            "mean_stim_ms": np.nanmean(stim_values)
            if len(data) else np.nan,
            "mean_nostim_ms": np.nanmean(nostim_values)
            if len(data) else np.nan,
            "mean_difference_stim_minus_nostim_ms": (
                np.nanmean(differences)
                if len(differences)
                else np.nan
            ),
            "median_difference_ms": (
                np.nanmedian(differences)
                if len(differences)
                else np.nan
            ),
            "sd_difference_ms": (
                np.nanstd(differences, ddof=1)
                if len(differences) >= 2
                else np.nan
            ),
            "cohens_dz": _paired_effect_size_dz(differences),
        }

        if len(data) >= 2:
            t_result = ttest_rel(
                stim_values,
                nostim_values,
                nan_policy="omit",
            )

            result["paired_t_statistic"] = t_result.statistic
            result["paired_t_pvalue"] = t_result.pvalue
        else:
            result["paired_t_statistic"] = np.nan
            result["paired_t_pvalue"] = np.nan

        if len(data) >= 2 and np.any(differences != 0):
            try:
                w_result = wilcoxon(
                    stim_values,
                    nostim_values,
                    zero_method="wilcox",
                    alternative="two-sided",
                )

                result["wilcoxon_statistic"] = w_result.statistic
                result["wilcoxon_pvalue"] = w_result.pvalue
            except ValueError:
                result["wilcoxon_statistic"] = np.nan
                result["wilcoxon_pvalue"] = np.nan
        else:
            result["wilcoxon_statistic"] = np.nan
            result["wilcoxon_pvalue"] = np.nan

        # Does the STIM–NOSTIM difference evolve over the session?
        if len(data) >= 3:
            trend = linregress(
                data["trial_num"].to_numpy(dtype=float),
                differences,
            )

            result["difference_slope_ms_per_trial"] = trend.slope
            result["difference_slope_r"] = trend.rvalue
            result["difference_slope_pvalue"] = trend.pvalue
        else:
            result["difference_slope_ms_per_trial"] = np.nan
            result["difference_slope_r"] = np.nan
            result["difference_slope_pvalue"] = np.nan

        rows.append(result)

    return pd.DataFrame(rows)


def compute_rt_block_summary(
    long_data: pd.DataFrame,
    rt_col: str = "reaction_time_ms",
) -> pd.DataFrame:
    """
    Summarize reaction time by participant, condition, ISI and
    global 80-trial block.
    """

    valid_data = long_data[
        long_data["reaction_time_valid"].astype(bool)
        & np.isfinite(long_data[rt_col])
    ].copy()

    summary = (
        valid_data
        .groupby(
            [
                "base_participant",
                "condition",
                "isi_bin",
                "global_block",
            ]
        )
        .agg(
            n_valid=(rt_col, "size"),
            mean_rt_ms=(rt_col, "mean"),
            median_rt_ms=(rt_col, "median"),
            sd_rt_ms=(rt_col, "std"),
            mean_centered_rt_ms=(
                "reaction_time_centered_ms",
                "mean",
            ),
            sd_centered_rt_ms=(
                "reaction_time_centered_ms",
                "std",
            ),
        )
        .reset_index()
    )

    total_counts = (
        long_data
        .groupby(
            [
                "base_participant",
                "condition",
                "isi_bin",
                "global_block",
            ]
        )
        .size()
        .rename("n_total")
        .reset_index()
    )

    summary = summary.merge(
        total_counts,
        on=[
            "base_participant",
            "condition",
            "isi_bin",
            "global_block",
        ],
        how="outer",
    )

    summary["n_invalid"] = (
        summary["n_total"] - summary["n_valid"].fillna(0)
    )

    summary["invalid_percentage"] = (
        100 * summary["n_invalid"] / summary["n_total"]
    )

    return summary


def compute_paired_block_statistics(
    paired_data: pd.DataFrame,
    difference_col: str = "rt_difference_centered_ms",
) -> pd.DataFrame:
    """
    Compute descriptive and paired statistics within each 80-trial block.
    """

    rows = []

    group_cols = [
        "base_participant",
        "isi_bin",
        "global_block",
    ]

    for group_values, group in paired_data.groupby(group_cols):

        base_participant, isi_bin, global_block = group_values

        valid = (
            group["pair_valid"].astype(bool)
            & np.isfinite(group[difference_col])
        )

        data = group.loc[valid]
        differences = data[difference_col].to_numpy(dtype=float)

        result = {
            "base_participant": base_participant,
            "isi_bin": isi_bin,
            "global_block": global_block,
            "n_valid_pairs": len(data),
            "mean_difference_stim_minus_nostim_ms": (
                np.mean(differences)
                if len(differences)
                else np.nan
            ),
            "median_difference_ms": (
                np.median(differences)
                if len(differences)
                else np.nan
            ),
            "sd_difference_ms": (
                np.std(differences, ddof=1)
                if len(differences) >= 2
                else np.nan
            ),
            "cohens_dz": _paired_effect_size_dz(differences),
        }

        if len(data) >= 2:
            stim = data[
                "reaction_time_centered_ms_stim"
            ].to_numpy(dtype=float)

            nostim = data[
                "reaction_time_centered_ms_nostim"
            ].to_numpy(dtype=float)

            t_result = ttest_rel(stim, nostim)

            result["paired_t_statistic"] = t_result.statistic
            result["paired_t_pvalue"] = t_result.pvalue

            if np.any(differences != 0):
                try:
                    w_result = wilcoxon(stim, nostim)
                    result["wilcoxon_statistic"] = w_result.statistic
                    result["wilcoxon_pvalue"] = w_result.pvalue
                except ValueError:
                    result["wilcoxon_statistic"] = np.nan
                    result["wilcoxon_pvalue"] = np.nan
            else:
                result["wilcoxon_statistic"] = np.nan
                result["wilcoxon_pvalue"] = np.nan
        else:
            result["paired_t_statistic"] = np.nan
            result["paired_t_pvalue"] = np.nan
            result["wilcoxon_statistic"] = np.nan
            result["wilcoxon_pvalue"] = np.nan

        rows.append(result)

    return pd.DataFrame(rows)

def compute_stim_nostim_trend_statistics(
    long_data,
    y_col="reaction_time_centered_ms",
    hac_maxlags=10,
):
    """
    Compare the temporal RT trends between STIM and NOSTIM.

    For each participant and ISI, fit:

        y ~ trial_num * condition

    NOSTIM is used as the reference condition.

    The interaction coefficient represents:

        slope_STIM - slope_NOSTIM

    Interpretation
    --------------
    interaction_slope_difference < 0:
        STIM decreases more strongly, or increases less strongly,
        than NOSTIM.

    interaction_slope_difference > 0:
        STIM increases more strongly, or decreases less strongly,
        than NOSTIM.

    HAC standard errors are used to reduce sensitivity to temporal
    autocorrelation and heteroscedasticity.
    """

    required_cols = {
        "base_participant",
        "condition",
        "isi_bin",
        "trial_num",
        "reaction_time_valid",
        y_col,
    }

    missing = required_cols.difference(long_data.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    rows = []

    for (participant, isi), group in long_data.groupby(
        ["base_participant", "isi_bin"]
    ):

        data = group[
            group["condition"].isin(["STIM", "NOSTIM"])
            & group["reaction_time_valid"].astype(bool)
            & np.isfinite(group[y_col])
            & np.isfinite(group["trial_num"])
        ].copy()

        n_stim = int((data["condition"] == "STIM").sum())
        n_nostim = int((data["condition"] == "NOSTIM").sum())

        result = {
            "base_participant": participant,
            "isi_bin": isi,
            "n_stim": n_stim,
            "n_nostim": n_nostim,
        }

        if (
            n_stim < 3
            or n_nostim < 3
            or data["trial_num"].nunique() < 3
        ):
            result.update(
                {
                    "nostim_intercept_ms": np.nan,
                    "stim_intercept_ms": np.nan,
                    "nostim_slope_ms_per_trial": np.nan,
                    "stim_slope_ms_per_trial": np.nan,
                    "interaction_slope_difference_ms_per_trial": np.nan,
                    "interaction_tvalue": np.nan,
                    "interaction_pvalue": np.nan,
                    "model_r_squared": np.nan,
                    "model_r_squared_adjusted": np.nan,
                }
            )

            rows.append(result)
            continue

        # Explicitly use NOSTIM as the reference condition
        formula = (
            f"{y_col} ~ trial_num * "
            'C(condition, Treatment(reference="NOSTIM"))'
        )

        model = smf.ols(
            formula=formula,
            data=data,
        ).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": hac_maxlags},
        )

        condition_term = (
            'C(condition, Treatment(reference="NOSTIM"))[T.STIM]'
        )

        interaction_term = (
            'trial_num:'
            'C(condition, Treatment(reference="NOSTIM"))[T.STIM]'
        )

        nostim_intercept = model.params.get(
            "Intercept",
            np.nan,
        )

        stim_intercept_difference = model.params.get(
            condition_term,
            np.nan,
        )

        nostim_slope = model.params.get(
            "trial_num",
            np.nan,
        )

        slope_difference = model.params.get(
            interaction_term,
            np.nan,
        )

        stim_intercept = (
            nostim_intercept + stim_intercept_difference
        )

        stim_slope = (
            nostim_slope + slope_difference
        )

        result.update(
            {
                "nostim_intercept_ms": nostim_intercept,
                "stim_intercept_ms": stim_intercept,
                "nostim_slope_ms_per_trial": nostim_slope,
                "stim_slope_ms_per_trial": stim_slope,
                "interaction_slope_difference_ms_per_trial": (
                    slope_difference
                ),
                "interaction_tvalue": model.tvalues.get(
                    interaction_term,
                    np.nan,
                ),
                "interaction_pvalue": model.pvalues.get(
                    interaction_term,
                    np.nan,
                ),
                "model_r_squared": model.rsquared,
                "model_r_squared_adjusted": model.rsquared_adj,
            }
        )

        rows.append(result)

    return pd.DataFrame(rows)

def compute_stim_nostim_variability_trend_statistics(
    long_data,
    variability_col="reaction_time_rolling_sd_ms",
    hac_maxlags=10,
):
    """
    Compare temporal trends in RT variability between STIM and NOSTIM.

    For each participant and ISI, fit:

        variability ~ trial_num * condition

    NOSTIM is the reference condition.

    The interaction coefficient is:

        slope_STIM - slope_NOSTIM

    Negative interaction:
        variability decreases more strongly under STIM.

    Positive interaction:
        variability increases more strongly under STIM.
    """

    required_cols = {
        "base_participant",
        "condition",
        "isi_bin",
        "trial_num",
        variability_col,
    }

    missing = required_cols.difference(long_data.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    rows = []

    for (participant, isi), group in long_data.groupby(
        ["base_participant", "isi_bin"]
    ):

        data = group[
            group["condition"].isin(["STIM", "NOSTIM"])
            & np.isfinite(group[variability_col])
            & np.isfinite(group["trial_num"])
        ].copy()

        n_stim = int((data["condition"] == "STIM").sum())
        n_nostim = int((data["condition"] == "NOSTIM").sum())

        result = {
            "base_participant": participant,
            "isi_bin": isi,
            "n_stim": n_stim,
            "n_nostim": n_nostim,
        }

        if (
            n_stim < 3
            or n_nostim < 3
            or data["trial_num"].nunique() < 3
        ):
            result.update(
                {
                    "nostim_slope_sd_ms_per_trial": np.nan,
                    "stim_slope_sd_ms_per_trial": np.nan,
                    "interaction_slope_difference_sd_ms_per_trial": np.nan,
                    "interaction_tvalue": np.nan,
                    "interaction_pvalue": np.nan,
                    "model_r_squared": np.nan,
                    "model_r_squared_adjusted": np.nan,
                }
            )
            rows.append(result)
            continue

        formula = (
            f"{variability_col} ~ trial_num * "
            'C(condition, Treatment(reference="NOSTIM"))'
        )

        model = smf.ols(
            formula=formula,
            data=data,
        ).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": hac_maxlags},
        )

        interaction_term = (
            'trial_num:'
            'C(condition, Treatment(reference="NOSTIM"))[T.STIM]'
        )

        nostim_slope = model.params.get(
            "trial_num",
            np.nan,
        )

        slope_difference = model.params.get(
            interaction_term,
            np.nan,
        )

        stim_slope = nostim_slope + slope_difference

        result.update(
            {
                "nostim_slope_sd_ms_per_trial": nostim_slope,
                "stim_slope_sd_ms_per_trial": stim_slope,
                "interaction_slope_difference_sd_ms_per_trial": slope_difference,
                "interaction_tvalue": model.tvalues.get(
                    interaction_term,
                    np.nan,
                ),
                "interaction_pvalue": model.pvalues.get(
                    interaction_term,
                    np.nan,
                ),
                "model_r_squared": model.rsquared,
                "model_r_squared_adjusted": model.rsquared_adj,
            }
        )

        rows.append(result)

    return pd.DataFrame(rows)