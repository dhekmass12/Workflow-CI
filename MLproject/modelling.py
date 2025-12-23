import warnings
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
warnings.simplefilter("ignore")
from xgboost import XGBClassifier
import dagshub
import sys
# dagshub.init(repo_owner='dhekmass12', repo_name='modelling_without_tuning', mlflow=True)

# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# mlflow.set_experiment("Modelling Without Tuning")


if __name__ == "__main__":
    df = pd.read_csv("dataset_preprocessing.csv")
    X = df.drop(['Failure Type'], axis=1)
    y = df['Failure Type']

    # Train/Val/Test: 80/10/10
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.11, random_state=42, stratify=y_train_val)

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

    os.environ.pop("MLFLOW_RUN_ID", None)

    with mlflow.start_run():
        mlflow.log_artifact("dataset_preprocessing.csv", artifact_path="datasets")
        mlflow.log_artifact("modelling.py", artifact_path="scripts")
        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train[0:5]
            )
