import pandas as pd


def clean_prediction_data(
    features: dict[str, float], feature_names: list[str]
) -> pd.DataFrame:
    missing = sorted(set(feature_names).difference(features))
    unexpected = sorted(set(features).difference(feature_names))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing features: {missing}")
        if unexpected:
            details.append(f"unexpected features: {unexpected}")
        raise ValueError("; ".join(details))
    return pd.DataFrame(
        [[features[name] for name in feature_names]], columns=feature_names
    )
