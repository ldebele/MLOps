import os
import pickle
import argparse 
import numpy as np

import mlflow 

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


mlflow.set_tracking_uri("postgresql://mlflow_user:user123@localhost:5432/mlflow_db")
mlflow.set_experiment("nyc-green-taxi-fine-tune-experiment")


def train(params):
    
    with mlflow.start_run():
        
        mlflow.log_params(params)
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        return {"loss": rmse, "status": STATUS_OK}


def run(num_trials):

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)

    fmin(
        fn=train,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def opt_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", 
        type=str, 
        required=True, 
        help="Location where the processed NYC taxi trip data was saved."
    )
    parser.add_argument(
        "--max_evals", 
        type=int, 
        default=50, 
        help="number of parameter evaluations for the optimizer to explore."
    )

    return parser.parse_args()



if __name__ == "__main__":

    args = opt_parser()

    X_train, y_train = load_pickle(os.path.join(args.source, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(args.source, "val.pkl"))

    run(args.max_evals)