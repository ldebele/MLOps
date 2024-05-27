import os 
import pickle
import argparse

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



MLFLOW_TRACKING_URI = "postgresql://mlflow_user:user123@localhost:5432/mlflow_db"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
# mlflow.sklearn.autolog()



def train_model(params):

    with mlflow.start_run():
        for param in RF_PARAMS:
            params[param] = int(params[param])

        mlflow.log_params(params)

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)

        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)
    


def run_register_model(top_n):

    client = MlflowClient()

    # # Retrieve the top_n model runs and log the models
    # experiment = client.get_experiment_by_name("nyc-green-taxi-fine-tune-experiment")
    # runs = client.search_runs(
    #                 experiment_ids=experiment.experiment_id,
    #                 run_view_type=ViewType.ACTIVE_ONLY,
    #                 max_results=top_n,
    #                 order_by=["metrics.rmse ASC"])
    
    # for run in runs:
    #     train_model(params=run.data.params)

    print("------------------")

    # select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_runs = client.search_runs(
                    experiment_ids=experiment.experiment_id,
                    run_view_type=ViewType.ACTIVE_ONLY,
                    max_results=top_n,
                    order_by=["metrics.rmse ASC"])[0]
    
    print(best_runs)
       
    # register the best model
    mlflow.register_model(
        model_uri=f"runs;/{best_runs.info.run_id}/models",
        name="green-taxi-rf"
    )



def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    

def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="Location where the processed NYC taxi trip data was saved.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top models that need to be evaluated to decide which one to promote.")

    return parser.parse_args()



if __name__ == "__main__":

    args = opt_parser()

    X_train, y_train = load_pickle(os.path.join(args.source, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(args.source, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(args.source, "test.pkl"))


    run_register_model(args.top_n)


    