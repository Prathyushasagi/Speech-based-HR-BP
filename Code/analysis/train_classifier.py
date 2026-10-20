import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from xgboost import XGBClassifier


# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv(
    "results/master_features_dataset.csv"
)

df = df.dropna()


# ---------------------------------------------------
# Create Balanced HR Classes
# ---------------------------------------------------

low_threshold = df["target_bpm"].quantile(0.33)

high_threshold = df["target_bpm"].quantile(0.66)

print(f"Low Threshold: {low_threshold:.2f}")

print(f"High Threshold: {high_threshold:.2f}")


def bpm_to_class(bpm):

    if bpm < low_threshold:
        return 0

    elif bpm < high_threshold:
        return 1

    else:
        return 2


df["hr_class"] = df["target_bpm"].apply(
    bpm_to_class
)


# ---------------------------------------------------
# Feature Selection
# ---------------------------------------------------

selected_features = [

    "rms_mean",
    "rms_std",

    "zcr_mean",
    "zcr_std",

    "spectral_centroid_mean",
    "spectral_centroid_std",

    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",

    "spectral_rolloff_mean",
    "spectral_rolloff_std",

    "pitch_mean",
    "pitch_std"
]



# ---------------------------------------------------
# Subject Independent Split
# ---------------------------------------------------

subjects = sorted(df["subject_id"].unique())

train_subjects = subjects[:60]

test_subjects = subjects[60:]


train_df = df[
    df["subject_id"].isin(train_subjects)
]

test_df = df[
    df["subject_id"].isin(test_subjects)
]


X_train = train_df[selected_features]

y_train = train_df["hr_class"]

X_test = test_df[selected_features]

y_test = test_df["hr_class"]


# ---------------------------------------------------
# Models
# ---------------------------------------------------

models = {

    "logistic":
        LogisticRegression(max_iter=5000),

    "random_forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "extra_trees":
        ExtraTreesClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "xgboost":
        XGBClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss"
        )
}


# ---------------------------------------------------
# Train + Evaluate
# ---------------------------------------------------

best_accuracy = 0
best_pipeline = None
best_model_name = None


for model_name, model in models.items():

    print(f"\nTraining {model_name}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")

    print("\nClassification Report")

    print(classification_report(
        y_test,
        preds
    ))

    if acc > best_accuracy:

        best_accuracy = acc

        best_pipeline = pipeline

        best_model_name = model_name


# ---------------------------------------------------
# Save Best Model
# ---------------------------------------------------

joblib.dump(
    best_pipeline,
    "results/best_hr_classifier.pkl"
)

print("\nBEST MODEL")

print(best_model_name)

print(f"Accuracy: {best_accuracy:.4f}")

print("\nModel saved successfully!")