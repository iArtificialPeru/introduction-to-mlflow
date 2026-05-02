import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Si ya configuraste la variable de entorno MLFLOW_TRACKING_URI, no es necesario establecerla aquí.
# en este caso la varaible de entorno fue
# export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
# mlflow.set_tracking_uri()  

mlflow.set_experiment("Diabetes_Prediction")
mlflow.sklearn.autolog()

# Cargar datos
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

# Iniciar experimento
with mlflow.start_run():
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(model, "model")

print("¡Ejecución registrada en MLflow y con Tracking Server SQLite local!")
