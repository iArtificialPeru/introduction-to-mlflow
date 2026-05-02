import mlflow
import mlflow.pyfunc
import pandas as pd

mlflow.set_experiment("lab_custom_model_demo")

class MyAdderModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return model_input.sum(axis=1)

with mlflow.start_run(run_name="custom_model_example") as run:
    mlflow.pyfunc.log_model(
        artifact_path="adder_model",
        python_model=MyAdderModel()
    )
    run_id = run.info.run_id

loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/adder_model")
df_test = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
print("Resultado:", loaded_model.predict(df_test))
