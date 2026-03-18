from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from nba_ou.config.settings import SETTINGS


@dataclass(frozen=True)
class PredictionModelDefinition:
    key: str
    label: str
    column_prefix: str
    aliases: tuple[str, ...]
    is_total_points: bool = True


_NUMBER_WORDS = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}


def _folder_name_from_prefix(prefix: str) -> str:
    prefix_path = Path(prefix.rstrip("/"))
    return prefix_path.parent.name if prefix_path.name == "production" else prefix_path.name


def _label_from_folder(folder_name: str) -> str:
    if folder_name == "total_points_full_dataset":
        return "Full Dataset"

    match = re.fullmatch(r"total_points_last_(\d+)_seasons", folder_name)
    if match:
        return f"{match.group(1)} Seasons"

    return folder_name.replace("_", " ").title()


def _column_prefix_from_folder(folder_name: str) -> str:
    if folder_name.startswith("total_points_"):
        return folder_name[len("total_points_") :]
    return folder_name


def _aliases_from_folder(folder_name: str) -> tuple[str, ...]:
    aliases: set[str] = {folder_name.lower()}

    if folder_name == "total_points_full_dataset":
        aliases.update(
            {
                "full_dataset_total_points",
                "full_xgb_total_points",
                "full_total_points",
            }
        )
        return tuple(sorted(aliases))

    match = re.fullmatch(r"total_points_last_(\d+)_seasons", folder_name)
    if match:
        n = match.group(1)
        word = _NUMBER_WORDS.get(n, n)
        aliases.update(
            {
                f"{word}_seasons_total_points",
                f"{word}_seasons_xgb_total_points",
                f"{n}_seasons_total_points",
                f"{n}_seasons_xgb_total_points",
                f"total_points_last_{n}_seasons",
            }
        )

    return tuple(sorted(aliases))


def get_prediction_model_definitions(
    *,
    include_tabpfn: bool = True,
) -> list[PredictionModelDefinition]:
    defs: list[PredictionModelDefinition] = []

    for configured_prefix in SETTINGS.prediction_model_prefixes:
        folder_name = _folder_name_from_prefix(configured_prefix)
        defs.append(
            PredictionModelDefinition(
                key=folder_name,
                label=_label_from_folder(folder_name),
                column_prefix=_column_prefix_from_folder(folder_name),
                aliases=_aliases_from_folder(folder_name),
                is_total_points=True,
            )
        )

    if include_tabpfn:
        defs.append(
            PredictionModelDefinition(
                key="TabPFNRegressor",
                label="TabPFN",
                column_prefix="tabpfn",
                aliases=(
                    "tabpfnregressor",
                    "tabpfn",
                    "tabpfn_client_regressor",
                ),
                is_total_points=True,
            )
        )

    return defs
