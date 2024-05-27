import os 
import pickle
import argparse
from datetime import datetime

import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


mlflow.set_tracking_uri("postgresql://mlflow_user:user123@localhost:5432/mlflow_db")
mlflow.set_experiment("nyc-green-taxi-experiment")



def run(source: str, model_path: str = "./models/"):

    X_train, y_train = load_pickle(os.path.join(source, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(source, "val.pkl"))

    with mlflow.start_run():

        mlflow.set_tag("developer", "Lemi")

        mlflow.autolog()

        max_depth = 10
        # mlflow.log_param("max_depth", max_depth)

        # train the model
        rf = RandomForestRegressor(max_depth=max_depth, random_state=0)
        rf.fit(X_train, y_train)

        # predict the validation
        y_pred = rf.predict(X_val)

        # calculate the rmse
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        # mlflow.log_metric("rmse", rmse)

        # save the model artifact
        model_name = f"rf_{str(datetime.now())}"
        model_path = os.path.join(model_path, model_name)
        with open(model_path, "wb") as f:
            pickle.dump(rf, f)

        # mlflow.log_artifact(local_path=model_path, artifact_path="models_pickle")
            

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
        

def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Location where the processed NYC taxi trip data was saved.")

    return parser.parse_args()



if __name__ == "__main__":

    args = opt_parser()

    run(args.source)