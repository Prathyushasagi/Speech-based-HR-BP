import time
import itertools
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


DATA_PATH = "results/master_features_dataset.csv"
OUTPUT_PATH = "results/feature_model_leaderboard.csv"


def load_data():
    df = pd.read_csv(DATA_PATH).dropna()
    return df


def build_feature_groups(df):
    return {
        "mfcc": [c for c in df.columns if c.startswith("mfcc_")],
        "delta_mfcc": [c for c in df.columns if c.startswith("delta_mfcc_")],
        "pitch": [c for c in ["pitch_mean", "pitch_std"] if c in df.columns],
        "energy": [c for c in ["rms_mean", "rms_std", "zcr_mean", "zcr_std"] if c in df.columns],
        "spectral": [c for c in df.columns if c.startswith("spectral_")],
    }


def build_feature_combinations(feature_groups):
    group_names = list(feature_groups.keys())
    combinations = []

    for r in range(1, len(group_names) + 1):
        for combo in itertools.combinations(group_names, r):
            columns = []
            for group in combo:
                columns.extend(feature_groups[group])

            combinations.append({
                "feature_set": "+".join(combo),
                "feature_groups": combo,
                "columns": columns,
                "num_features": len(columns),
            })

    return combinations


def build_models():
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=42
        ),
        "xgboost": XGBRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            verbosity=0
        ),
    }


def subject_split(df, train_ratio=0.8):
    subjects = sorted(df["subject_id"].unique())
    split_index = int(len(subjects) * train_ratio)

    train_subjects = subjects[:split_index]
    test_subjects = subjects[split_index:]

    train_df = df[df["subject_id"].isin(train_subjects)]
    test_df = df[df["subject_id"].isin(test_subjects)]

    return train_df, test_df, train_subjects, test_subjects


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def run_pipeline():
    df = load_data()

    print(f"Loaded dataset: {len(df)} samples")
    print(f"Subjects: {df['subject_id'].nunique()}")

    feature_groups = build_feature_groups(df)

    print("\nFeature groups:")
    for name, cols in feature_groups.items():
        print(f"{name}: {len(cols)} features")

    feature_combinations = build_feature_combinations(feature_groups)
    models = build_models()

    train_df, test_df, train_subjects, test_subjects = subject_split(df)

    print(f"\nTrain subjects: {len(train_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    results = []
    total_runs = len(feature_combinations) * len(models)
    run_number = 0

    for feature_combo in feature_combinations:
        feature_set = feature_combo["feature_set"]
        columns = feature_combo["columns"]

        X_train = train_df[columns]
        y_train = train_df["target_bpm"]

        X_test = test_df[columns]
        y_test = test_df["target_bpm"]

        for model_name, model in models.items():
            run_number += 1
            start = time.time()

            print(f"\nRun {run_number}/{total_runs}")
            print(f"Features: {feature_set}")
            print(f"Model: {model_name}")

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])

            try:
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)

                metrics = evaluate(y_test, preds)
                runtime = round(time.time() - start, 2)

                result = {
                    "feature_set": feature_set,
                    "model": model_name,
                    "num_features": feature_combo["num_features"],
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                    "runtime_sec": runtime,
                }

                results.append(result)

                print(f"RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, R2: {metrics['R2']:.4f}")

            except Exception as e:
                results.append({
                    "feature_set": feature_set,
                    "model": model_name,
                    "num_features": feature_combo["num_features"],
                    "MAE": None,
                    "RMSE": None,
                    "R2": None,
                    "runtime_sec": None,
                    "error": str(e),
                })

                print(f"Failed: {e}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="RMSE", ascending=True)

    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n==============================")
    print("TOP 10 PIPELINES")
    print("==============================")
    print(results_df.head(10))

    print(f"\nSaved leaderboard to: {OUTPUT_PATH}")

    best = results_df.iloc[0]

    print("\nBEST PIPELINE")
    print(f"Features: {best['feature_set']}")
    print(f"Model: {best['model']}")
    print(f"MAE: {best['MAE']:.4f}")
    print(f"RMSE: {best['RMSE']:.4f}")
    print(f"R2: {best['R2']:.4f}")


if __name__ == "__main__":
    run_pipeline()