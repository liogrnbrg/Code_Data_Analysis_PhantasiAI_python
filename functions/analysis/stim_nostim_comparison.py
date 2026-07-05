from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress, ttest_rel, wilcoxon
import statsmodels.formula.api as smf

from utils.conditions import (
    add_condition_columns,
    get_comparison_pairs,
    session_sort_key,
)


def _paired_effect_size_dz(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]

    if len(differences) < 2:
        return np.nan

    sd_difference = np.std(differences, ddof=1)

    if not np.isfinite(sd_difference) or sd_difference == 0:
        return np.nan

    return np.mean(differences) / sd_difference


def prepare_rt_stim_nostim_comparison(
    rt_data: pd.DataFrame,
    rt_col: str = "reaction_time_ms",
    n_baseline_trials: int = 10,
    config=None,
    comparison_pairs=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare long and paired condition-comparison datasets.

    Despite the old function name, this now supports:
        NOSTIM
        FIXED_STIM
        STIM

    It creates paired tables for every comparison defined in config.
    Missing conditions are skipped automatically.
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

    if comparison_pairs is None:
        comparison_pairs = get_comparison_pairs(config)

    long_data = rt_data.copy()

    long_data = add_condition_columns(
        long_data,
        config=config,
        participant_col="participant_id",
    )

    long_data = long_data[
        long_data["condition"] != "UNKNOWN"
    ].copy()

    if long_data.empty:
        raise ValueError(
            "No recognized condition sessions were found. "
            "Expected names like X_NOSTIM, X_FIXED_STIM, X_STIM, X_STIM_2."
        )

    long_data["session_id"] = long_data["participant_id"].astype(str)

    long_data["rt_valid_value"] = long_data[rt_col].where(
        long_data["reaction_time_valid"].astype(bool)
        & np.isfinite(long_data[rt_col])
    )

    group_cols = [
        "base_participant",
        "participant_id",
        "condition",
        "session_number",
        "isi_bin",
    ]

    baseline_rows = (
        long_data[
            long_data["rt_valid_value"].notna()
        ]
        .sort_values(group_cols + ["trial_num"])
        .groupby(
            group_cols,
            group_keys=False,
            dropna=False,
        )
        .head(n_baseline_trials)
    )

    baselines = (
        baseline_rows
        .groupby(
            group_cols,
            dropna=False,
        )["rt_valid_value"]
        .agg(
            rt_baseline_ms="mean",
            n_baseline_valid="size",
        )
        .reset_index()
    )

    long_data = long_data.merge(
        baselines,
        on=group_cols,
        how="left",
        validate="many_to_one",
    )

    long_data["reaction_time_centered_ms"] = (
        long_data["rt_valid_value"]
        - long_data["rt_baseline_ms"]
    )

    long_data["global_block"] = (
        (long_data["trial_num"].astype(int) - 1) // 80
    ) + 1

    paired_tables = []

    merge_cols = [
        "trial_num",
        "isi_bin",
    ]

    value_cols = [
        rt_col,
        "reaction_time_centered_ms",
        "reaction_time_valid",
    ]

    for base_participant, participant_data in long_data.groupby(
        "base_participant",
        sort=False,
    ):
        for comparison in comparison_pairs:
            test_condition = comparison["test_condition"]
            reference_condition = comparison["reference_condition"]
            comparison_name = comparison["comparison_name"]
            comparison_label = comparison.get(
                "comparison_label",
                f"{test_condition} − {reference_condition}",
            )

            test_sessions = sorted(
                participant_data.loc[
                    participant_data["condition"] == test_condition,
                    "participant_id",
                ]
                .dropna()
                .astype(str)
                .unique(),
                key=lambda x: session_sort_key(x, config=config),
            )

            reference_sessions = sorted(
                participant_data.loc[
                    participant_data["condition"] == reference_condition,
                    "participant_id",
                ]
                .dropna()
                .astype(str)
                .unique(),
                key=lambda x: session_sort_key(x, config=config),
            )

            if len(test_sessions) == 0 or len(reference_sessions) == 0:
                continue

            for reference_session in reference_sessions:
                reference = participant_data[
                    participant_data["participant_id"]
                    == reference_session
                ][merge_cols + value_cols].copy()

                reference = reference.rename(
                    columns={
                        col: f"{col}_reference"
                        for col in value_cols
                    }
                )

                for test_session in test_sessions:
                    test = participant_data[
                        participant_data["participant_id"]
                        == test_session
                    ][merge_cols + value_cols].copy()

                    test = test.rename(
                        columns={
                            col: f"{col}_test"
                            for col in value_cols
                        }
                    )

                    paired_session = test.merge(
                        reference,
                        on=merge_cols,
                        how="outer",
                        validate="one_to_one",
                    )

                    paired_session["base_participant"] = base_participant
                    paired_session["comparison_name"] = comparison_name
                    paired_session["comparison_label"] = comparison_label
                    paired_session["test_condition"] = test_condition
                    paired_session["reference_condition"] = reference_condition
                    paired_session["test_session_id"] = test_session
                    paired_session["reference_session_id"] = reference_session

                    paired_tables.append(paired_session)

    if paired_tables:
        paired = pd.concat(
            paired_tables,
            ignore_index=True,
        )
    else:
        paired = pd.DataFrame()

    required_paired_cols = [
        f"{rt_col}_test",
        f"{rt_col}_reference",
        "reaction_time_centered_ms_test",
        "reaction_time_centered_ms_reference",
        "reaction_time_valid_test",
        "reaction_time_valid_reference",
    ]

    for col in required_paired_cols:
        if col not in paired.columns:
            paired[col] = np.nan

    test_valid = (
        paired["reaction_time_valid_test"]
        .fillna(False)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    reference_valid = (
        paired["reaction_time_valid_reference"]
        .fillna(False)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    paired["pair_valid"] = (
        test_valid
        & reference_valid
        & np.isfinite(paired[f"{rt_col}_test"])
        & np.isfinite(paired[f"{rt_col}_reference"])
    )

    paired["rt_difference_raw_ms"] = (
        paired[f"{rt_col}_test"]
        - paired[f"{rt_col}_reference"]
    )

    paired["rt_difference_centered_ms"] = (
        paired["reaction_time_centered_ms_test"]
        - paired["reaction_time_centered_ms_reference"]
    )

    if not paired.empty:
        paired["global_block"] = (
            (paired["trial_num"].astype(int) - 1) // 80
        ) + 1

    return long_data, paired


def add_rolling_rt_variability(
    long_data: pd.DataFrame,
    rt_col: str = "reaction_time_ms",
    rolling_window: int = 10,
    min_periods: int = 5,
) -> pd.DataFrame:
    data = long_data.copy()

    output_col = "reaction_time_rolling_sd_ms"
    data[output_col] = np.nan

    group_cols = [
        "base_participant",
        "participant_id",
        "condition",
        "isi_bin",
    ]

    for _, group_index in data.groupby(group_cols, dropna=False).groups.items():
        group = data.loc[group_index].sort_values("trial_num")

        valid_rt = group[rt_col].where(
            group["reaction_time_valid"].astype(bool)
            & np.isfinite(group[rt_col])
        )

        rolling_sd = (
            valid_rt
            .rolling(
                window=rolling_window,
                min_periods=min_periods,
                center=True,
            )
            .std()
        )

        data.loc[group.index, output_col] = rolling_sd

    return data


def prepare_variability_pairs(
    long_data: pd.DataFrame,
    config=None,
    comparison_pairs=None,
) -> pd.DataFrame:
    variability_col = "reaction_time_rolling_sd_ms"

    if variability_col not in long_data.columns:
        raise KeyError(
            f"{variability_col} not found. "
            "Run add_rolling_rt_variability first."
        )

    if comparison_pairs is None:
        comparison_pairs = get_comparison_pairs(config)

    paired_tables = []

    merge_cols = [
        "trial_num",
        "isi_bin",
        "global_block",
    ]

    for base_participant, participant_data in long_data.groupby(
        "base_participant",
        sort=False,
    ):
        for comparison in comparison_pairs:
            test_condition = comparison["test_condition"]
            reference_condition = comparison["reference_condition"]
            comparison_name = comparison["comparison_name"]
            comparison_label = comparison.get(
                "comparison_label",
                f"{test_condition} − {reference_condition}",
            )

            test_sessions = sorted(
                participant_data.loc[
                    participant_data["condition"] == test_condition,
                    "participant_id",
                ]
                .dropna()
                .astype(str)
                .unique(),
                key=lambda x: session_sort_key(x, config=config),
            )

            reference_sessions = sorted(
                participant_data.loc[
                    participant_data["condition"] == reference_condition,
                    "participant_id",
                ]
                .dropna()
                .astype(str)
                .unique(),
                key=lambda x: session_sort_key(x, config=config),
            )

            if len(test_sessions) == 0 or len(reference_sessions) == 0:
                continue

            for reference_session in reference_sessions:
                reference = participant_data[
                    participant_data["participant_id"]
                    == reference_session
                ][merge_cols + [variability_col]].copy()

                reference = reference.rename(
                    columns={
                        variability_col: f"{variability_col}_reference"
                    }
                )

                for test_session in test_sessions:
                    test = participant_data[
                        participant_data["participant_id"]
                        == test_session
                    ][merge_cols + [variability_col]].copy()

                    test = test.rename(
                        columns={
                            variability_col: f"{variability_col}_test"
                        }
                    )

                    paired_session = test.merge(
                        reference,
                        on=merge_cols,
                        how="outer",
                        validate="one_to_one",
                    )

                    paired_session["base_participant"] = base_participant
                    paired_session["comparison_name"] = comparison_name
                    paired_session["comparison_label"] = comparison_label
                    paired_session["test_condition"] = test_condition
                    paired_session["reference_condition"] = reference_condition
                    paired_session["test_session_id"] = test_session
                    paired_session["reference_session_id"] = reference_session

                    paired_tables.append(paired_session)

    if not paired_tables:
        return pd.DataFrame()

    paired = pd.concat(
        paired_tables,
        ignore_index=True,
    )

    test_col = f"{variability_col}_test"
    reference_col = f"{variability_col}_reference"

    paired["variability_pair_valid"] = (
        np.isfinite(paired[test_col])
        & np.isfinite(paired[reference_col])
    )

    paired["rt_variability_difference_ms"] = (
        paired[test_col]
        - paired[reference_col]
    )

    return paired


def compute_rt_paired_statistics(
    paired_data: pd.DataFrame,
    difference_col: str = "rt_difference_centered_ms",
) -> pd.DataFrame:
    rows = []

    if paired_data.empty:
        return pd.DataFrame()

    group_cols = [
        "comparison_name",
        "comparison_label",
        "base_participant",
        "test_condition",
        "reference_condition",
        "test_session_id",
        "reference_session_id",
        "isi_bin",
    ]

    if difference_col == "rt_difference_centered_ms":
        test_col = "reaction_time_centered_ms_test"
        reference_col = "reaction_time_centered_ms_reference"
    else:
        test_col = "reaction_time_ms_test"
        reference_col = "reaction_time_ms_reference"

    for group_values, group in paired_data.groupby(
        group_cols,
        dropna=False,
    ):
        group_dict = dict(zip(group_cols, group_values))

        valid = (
            group["pair_valid"].astype(bool)
            & np.isfinite(group[difference_col])
        )

        data = group.loc[valid].sort_values("trial_num")

        differences = data[difference_col].to_numpy(dtype=float)

        test_values = data[test_col].to_numpy(dtype=float)
        reference_values = data[reference_col].to_numpy(dtype=float)

        result = {
            **group_dict,
            "n_valid_pairs": len(data),
            "mean_test_ms": np.nanmean(test_values) if len(data) else np.nan,
            "mean_reference_ms": np.nanmean(reference_values) if len(data) else np.nan,
            "mean_difference_test_minus_reference_ms": (
                np.nanmean(differences) if len(differences) else np.nan
            ),
            "median_difference_ms": (
                np.nanmedian(differences) if len(differences) else np.nan
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
                test_values,
                reference_values,
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
                    test_values,
                    reference_values,
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
    group_cols = [
        "base_participant",
        "participant_id",
        "condition",
        "condition_label",
        "isi_bin",
        "global_block",
    ]

    valid_data = long_data[
        long_data["reaction_time_valid"].astype(bool)
        & np.isfinite(long_data[rt_col])
    ].copy()

    summary = (
        valid_data
        .groupby(
            group_cols,
            dropna=False,
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
            group_cols,
            dropna=False,
        )
        .size()
        .rename("n_total")
        .reset_index()
    )

    summary = summary.merge(
        total_counts,
        on=group_cols,
        how="outer",
        validate="one_to_one",
    )

    summary["n_valid"] = summary["n_valid"].fillna(0)

    summary["n_invalid"] = (
        summary["n_total"]
        - summary["n_valid"]
    )

    summary["invalid_percentage"] = (
        100
        * summary["n_invalid"]
        / summary["n_total"]
    )

    return summary


def compute_paired_block_statistics(
    paired_data: pd.DataFrame,
    difference_col: str = "rt_difference_centered_ms",
) -> pd.DataFrame:
    rows = []

    if paired_data.empty:
        return pd.DataFrame()

    group_cols = [
        "comparison_name",
        "comparison_label",
        "base_participant",
        "test_condition",
        "reference_condition",
        "test_session_id",
        "reference_session_id",
        "isi_bin",
        "global_block",
    ]

    for group_values, group in paired_data.groupby(
        group_cols,
        dropna=False,
    ):
        group_dict = dict(zip(group_cols, group_values))

        valid = (
            group["pair_valid"].astype(bool)
            & np.isfinite(group[difference_col])
        )

        data = group.loc[valid].copy()

        differences = data[difference_col].to_numpy(dtype=float)

        result = {
            **group_dict,
            "n_valid_pairs": len(data),
            "mean_difference_test_minus_reference_ms": (
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
            test = data[
                "reaction_time_centered_ms_test"
            ].to_numpy(dtype=float)

            reference = data[
                "reaction_time_centered_ms_reference"
            ].to_numpy(dtype=float)

            t_result = ttest_rel(
                test,
                reference,
                nan_policy="omit",
            )

            result["paired_t_statistic"] = t_result.statistic
            result["paired_t_pvalue"] = t_result.pvalue

            if np.any(differences != 0):
                try:
                    w_result = wilcoxon(
                        test,
                        reference,
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
        else:
            result["paired_t_statistic"] = np.nan
            result["paired_t_pvalue"] = np.nan
            result["wilcoxon_statistic"] = np.nan
            result["wilcoxon_pvalue"] = np.nan

        rows.append(result)

    return pd.DataFrame(rows)


def _compute_condition_trend_statistics(
    long_data,
    y_col,
    value_label,
    hac_maxlags=10,
    config=None,
    comparison_pairs=None,
    valid_col="reaction_time_valid",
):
    if comparison_pairs is None:
        comparison_pairs = get_comparison_pairs(config)

    required_cols = {
        "base_participant",
        "participant_id",
        "condition",
        "isi_bin",
        "trial_num",
        y_col,
    }

    if valid_col is not None:
        required_cols.add(valid_col)

    missing = required_cols.difference(long_data.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    rows = []

    for base_participant, participant_data in long_data.groupby(
        "base_participant",
        sort=False,
    ):
        for comparison in comparison_pairs:
            test_condition = comparison["test_condition"]
            reference_condition = comparison["reference_condition"]
            comparison_name = comparison["comparison_name"]
            comparison_label = comparison.get(
                "comparison_label",
                f"{test_condition} − {reference_condition}",
            )

            test_sessions = sorted(
                participant_data.loc[
                    participant_data["condition"] == test_condition,
                    "participant_id",
                ]
                .dropna()
                .astype(str)
                .unique(),
                key=lambda x: session_sort_key(x, config=config),
            )

            reference_sessions = sorted(
                participant_data.loc[
                    participant_data["condition"] == reference_condition,
                    "participant_id",
                ]
                .dropna()
                .astype(str)
                .unique(),
                key=lambda x: session_sort_key(x, config=config),
            )

            if len(test_sessions) == 0 or len(reference_sessions) == 0:
                continue

            for reference_session in reference_sessions:
                for test_session in test_sessions:
                    pair_data = participant_data[
                        participant_data["participant_id"].isin(
                            [
                                reference_session,
                                test_session,
                            ]
                        )
                    ].copy()

                    pair_data["pair_condition"] = np.where(
                        pair_data["participant_id"] == reference_session,
                        "reference",
                        "test",
                    )

                    for isi_bin, group in pair_data.groupby(
                        "isi_bin",
                        dropna=False,
                    ):
                        valid = (
                            np.isfinite(group[y_col])
                            & np.isfinite(group["trial_num"])
                        )

                        if valid_col is not None:
                            valid = valid & group[valid_col].astype(bool)

                        data = group[valid].copy()

                        n_test = int(
                            (data["pair_condition"] == "test").sum()
                        )

                        n_reference = int(
                            (data["pair_condition"] == "reference").sum()
                        )

                        result = {
                            "comparison_name": comparison_name,
                            "comparison_label": comparison_label,
                            "base_participant": base_participant,
                            "test_condition": test_condition,
                            "reference_condition": reference_condition,
                            "test_session_id": test_session,
                            "reference_session_id": reference_session,
                            "isi_bin": isi_bin,
                            "n_test": n_test,
                            "n_reference": n_reference,
                        }

                        if (
                            n_test < 3
                            or n_reference < 3
                            or data["trial_num"].nunique() < 3
                        ):
                            result.update(
                                {
                                    f"reference_slope_{value_label}_per_trial": np.nan,
                                    f"test_slope_{value_label}_per_trial": np.nan,
                                    f"interaction_slope_difference_{value_label}_per_trial": np.nan,
                                    "interaction_tvalue": np.nan,
                                    "interaction_pvalue": np.nan,
                                    "model_r_squared": np.nan,
                                    "model_r_squared_adjusted": np.nan,
                                }
                            )

                            rows.append(result)
                            continue

                        formula = (
                            f"{y_col} ~ trial_num * "
                            'C(pair_condition, Treatment(reference="reference"))'
                        )

                        model = smf.ols(
                            formula=formula,
                            data=data,
                        ).fit(
                            cov_type="HAC",
                            cov_kwds={
                                "maxlags": hac_maxlags,
                            },
                        )

                        interaction_term = (
                            "trial_num:"
                            'C(pair_condition, Treatment(reference="reference"))'
                            "[T.test]"
                        )

                        condition_term = (
                            'C(pair_condition, Treatment(reference="reference"))'
                            "[T.test]"
                        )

                        reference_intercept = model.params.get(
                            "Intercept",
                            np.nan,
                        )

                        test_intercept_difference = model.params.get(
                            condition_term,
                            np.nan,
                        )

                        reference_slope = model.params.get(
                            "trial_num",
                            np.nan,
                        )

                        slope_difference = model.params.get(
                            interaction_term,
                            np.nan,
                        )

                        result.update(
                            {
                                f"reference_intercept_{value_label}": reference_intercept,
                                f"test_intercept_{value_label}": (
                                    reference_intercept
                                    + test_intercept_difference
                                ),
                                f"reference_slope_{value_label}_per_trial": reference_slope,
                                f"test_slope_{value_label}_per_trial": (
                                    reference_slope
                                    + slope_difference
                                ),
                                f"interaction_slope_difference_{value_label}_per_trial": slope_difference,
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


def compute_stim_nostim_trend_statistics(
    long_data,
    y_col="reaction_time_centered_ms",
    hac_maxlags=10,
    config=None,
):
    return _compute_condition_trend_statistics(
        long_data=long_data,
        y_col=y_col,
        value_label="ms",
        hac_maxlags=hac_maxlags,
        config=config,
        valid_col="reaction_time_valid",
    )


def compute_stim_nostim_variability_trend_statistics(
    long_data,
    variability_col="reaction_time_rolling_sd_ms",
    hac_maxlags=10,
    config=None,
):
    return _compute_condition_trend_statistics(
        long_data=long_data,
        y_col=variability_col,
        value_label="sd_ms",
        hac_maxlags=hac_maxlags,
        config=config,
        valid_col=None,
    )