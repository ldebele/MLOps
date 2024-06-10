import pathlib
import pickle 

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import mlflow

from prefect import task 



@task(log_prints=True)
def train(X_train: scipy.sparse._csr.csr_matrix,
          X_val: scipy.sparse._csr.csr_matrix,
          y_train: np.ndarray,
          y_val: np.ndarray,
          dv: sklearn.feature_extraction.DictVectorizer) -> None:
    """Train a model with best hyperparams."""

    with mlflow.start_run():

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        intercept = model.intercept_
        mlflow.log_metric("intercept", intercept)

        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/lr_preprocessor.b", "wb") as f:
            pickle.dump(dv, f)

        mlflow.log_artifact("models/lr_preprocessor.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(model, artifact_path="models_mlflow")


    return None




