import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Entrenamiento y log de un modelo en MLflow
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
clf = RandomForestClassifier(n_estimators=10, random_state=42)

experiment_name = "RandomForestClassifier_IrisModel"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="rf_model_registry_demo") as run:
    clf.fit(X_train, y_train)
    input_example = X_train[:5]
    # Inferir la firma del modelo
    signature = infer_signature(X_train, clf.predict(X_train))
    mlflow.sklearn.log_model(
        clf,
        "model",
        input_example=input_example,
        signature=signature
    )
    run_id = run.info.run_id
