import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge
)

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)

from xgboost import XGBRegressor




def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    start_time = time.time()

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    end_time = time.time()

    print(f"{name} completed in {end_time - start_time:.2f} sec")

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Time_sec": round(end_time - start_time, 2)
    }




if __name__ == "__main__":

    print("\nLoading dataset...")

    df = pd.read_csv(
        "results/master_features_dataset.csv"
    )

    print(f"Dataset loaded successfully!")

    print(f"Total Samples: {len(df)}")

    print(f"Total Subjects: {df['subject_id'].nunique()}")


    

    unique_subjects = sorted(
        df["subject_id"].unique()
    )

    train_subjects = unique_subjects[:60]

    test_subjects = unique_subjects[60:]

    print(f"\nTrain Subjects: {len(train_subjects)}")

    print(f"Test Subjects: {len(test_subjects)}")


    train_df = df[
        df["subject_id"].isin(train_subjects)
    ]

    test_df = df[
        df["subject_id"].isin(test_subjects)
    ]

    print(f"\nTrain Samples: {len(train_df)}")

    print(f"Test Samples: {len(test_df)}")


   

    X_train = train_df.drop(
        columns=["target_bpm", "subject_id"]
    )

    y_train = train_df["target_bpm"]

    X_test = test_df.drop(
        columns=["target_bpm", "subject_id"]
    )

    y_test = test_df["target_bpm"]

    print(f"\nFeature Count: {X_train.shape[1]}")


   

    models = {

        "Linear Regression":
            LinearRegression(),

        "Ridge Regression":
            Ridge(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=30,
                random_state=42,
                n_jobs=-1
            ),

        "Extra Trees":
            ExtraTreesRegressor(
                n_estimators=30,
                random_state=42,
                n_jobs=-1
            ),

        "XGBoost":
            XGBRegressor(
                n_estimators=30,
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror",
                verbosity=0
            )
    }



    results = []

    print("\nStarting benchmarking...")

    for name, model in models.items():

        metrics = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(metrics)


   

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="RMSE"
    )

  
    print("MODEL LEADERBOARD")


    print(results_df)



    results_df.to_csv(
        "results/model_results.csv",
        index=False
    )

    print("\nResults saved successfully!")