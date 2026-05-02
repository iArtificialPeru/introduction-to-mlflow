import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from mlflow.models.signature import infer_signature

mlflow.set_experiment("lab_model_signatures_demo")

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier().fit(X, y)
signature = infer_signature(X, clf.predict(X))

with mlflow.start_run(run_name="model_signatures_example"):
    mlflow.sklearn.log_model(clf, "rf_model", signature=signature)

print("Modelo guardado con signature inferida.")
