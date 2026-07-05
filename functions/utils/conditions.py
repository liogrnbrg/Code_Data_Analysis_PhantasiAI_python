from __future__ import annotations

import re
import pandas as pd


def get_condition_order(config=None):
    if config is None:
        config = {}

    return (
        config
        .get("conditions", {})
        .get(
            "condition_order",
            ["NOSTIM", "FIXED_STIM", "STIM"],
        )
    )


def get_condition_labels(config=None):
    if config is None:
        config = {}

    return (
        config
        .get("conditions", {})
        .get("condition_labels", {})
    )


def get_comparison_pairs(config=None):
    if config is None:
        config = {}

    return (
        config
        .get("conditions", {})
        .get(
            "comparison_pairs",
            [
                {
                    "comparison_name": "stim_vs_nostim",
                    "test_condition": "STIM",
                    "reference_condition": "NOSTIM",
                    "comparison_label": "STIM − NOSTIM",
                },
                {
                    "comparison_name": "fixed_stim_vs_nostim",
                    "test_condition": "FIXED_STIM",
                    "reference_condition": "NOSTIM",
                    "comparison_label": "FIXED_STIM − NOSTIM",
                },
                {
                    "comparison_name": "stim_vs_fixed_stim",
                    "test_condition": "STIM",
                    "reference_condition": "FIXED_STIM",
                    "comparison_label": "STIM − FIXED_STIM",
                },
            ],
        )
    )


def parse_session_id(participant_id, config=None):
    """
    Parse session identifiers such as:

        Lio_NOSTIM
        Lio_STIM
        Lio_STIM_2
        Lio_FIXED_STIM
        Lio_FIXED_STIM_2
        Yosra_STIM

    Returns:
        base_participant, condition, session_number
    """

    participant_id = str(participant_id)

    condition_order = get_condition_order(config)

    # Longest first is critical:
    # FIXED_STIM must be matched before STIM.
    conditions = sorted(
        condition_order,
        key=len,
        reverse=True,
    )

    for condition in conditions:
        pattern = (
            rf"^(?P<base>.+)_"
            rf"{re.escape(condition)}"
            rf"(?:_(?P<number>\d+))?$"
        )

        match = re.fullmatch(
            pattern,
            participant_id,
        )

        if match is not None:
            number = match.group("number")

            return {
                "participant_id": participant_id,
                "base_participant": match.group("base"),
                "condition": condition,
                "session_number": int(number) if number else 1,
            }

    return {
        "participant_id": participant_id,
        "base_participant": participant_id,
        "condition": "UNKNOWN",
        "session_number": 1,
    }


def add_condition_columns(
    data,
    config=None,
    participant_col="participant_id",
):
    data = data.copy()

    parsed = [
        parse_session_id(value, config=config)
        for value in data[participant_col]
    ]

    parsed_df = pd.DataFrame(parsed, index=data.index)

    data["base_participant"] = parsed_df["base_participant"]
    data["condition"] = parsed_df["condition"]
    data["session_number"] = parsed_df["session_number"]

    condition_labels = get_condition_labels(config)

    data["condition_label"] = (
        data["condition"]
        .map(condition_labels)
        .fillna(data["condition"])
    )

    return data


def condition_sort_key(condition, config=None):
    condition_order = get_condition_order(config)

    try:
        return (
            condition_order.index(condition),
            str(condition),
        )
    except ValueError:
        return (
            len(condition_order),
            str(condition),
        )


def session_sort_key(participant_id, config=None):
    parsed = parse_session_id(
        participant_id,
        config=config,
    )

    return (
        str(parsed["base_participant"]),
        condition_sort_key(parsed["condition"], config=config),
        parsed["session_number"],
        str(participant_id),
    )