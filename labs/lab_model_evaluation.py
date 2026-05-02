import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

mlflow.set_experiment("lab_model_evaluation_demo")

X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = RandomForestClassifier().fit(X_train, y_train)

df_test = pd.DataFrame(X_test, columns=feature_names)
df_test["target"] = y_test

with mlflow.start_run(run_name="model_evaluation_example") as run:
    mlflow.sklearn.log_model(clf, "rf_model")
    run_id = run.info.run_id
    eval_results = mlflow.evaluate(
        model=f"runs:/{run_id}/rf_model",
        data=df_test,
        targets="target",
        model_type="classifier"
    )
    print("Métricas:", eval_results.metrics)
