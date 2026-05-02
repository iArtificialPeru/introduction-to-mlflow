import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

mlflow.set_experiment("lab_input_examples_demo")

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier().fit(X, y)
input_example = X[:2]

with mlflow.start_run(run_name="input_examples_example"):
    mlflow.sklearn.log_model(clf, "rf_model", input_example=input_example)

print("Modelo guardado con input_example.")
