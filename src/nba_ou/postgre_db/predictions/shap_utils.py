from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShapFeatureContribution:
    feature: str
    value: float


def parse_serialized_shap_contributions(
    raw_value: object,
) -> list[ShapFeatureContribution]:
    """
    Parse stored SHAP text into typed feature/value pairs.

    Expected format:
        FEATURE_A:+0.321,FEATURE_B:-0.123
    """
    if raw_value is None:
        return []

    text = str(raw_value).strip()
    if not text or text.lower() == "nan":
        return []

    contributions: list[ShapFeatureContribution] = []
    for item in text.split(","):
        chunk = item.strip()
        if not chunk or ":" not in chunk:
            continue

        feature, value_text = chunk.rsplit(":", 1)
        feature = feature.strip()
        value_text = value_text.strip()
        if not feature or not value_text:
            continue

        try:
            value = float(value_text)
        except ValueError:
            continue

        contributions.append(ShapFeatureContribution(feature=feature, value=value))

    return contributions
