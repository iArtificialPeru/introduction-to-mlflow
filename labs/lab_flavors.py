import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

mlflow.set_experiment("lab_flavors_demo")

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier().fit(X, y)

with mlflow.start_run(run_name="flavor_example"):
    mlflow.sklearn.log_model(clf, "rf_model")

print("Modelo guardado con flavors: 'sklearn' y 'python_function'.")
