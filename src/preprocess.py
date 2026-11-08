import numpy as np

from breast_cancer_pipeline import PROJECT_ROOT, load_config, load_dataset, prepare_splits


def run_preprocessing():
    config = load_config()
    frame, feature_columns = load_dataset(config)
    X_train, X_test, y_train, y_test, _, _ = prepare_splits(
        config, frame, feature_columns
    )
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    np.save(processed_dir / "y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save(processed_dir / "y_test.npy", y_test.to_numpy(dtype=np.int64))
    print(
        f"[SUCCESS] Prepared {len(X_train)} training and {len(X_test)} test rows "
        f"with {len(feature_columns)} numeric features."
    )


if __name__ == "__main__":
    run_preprocessing()
