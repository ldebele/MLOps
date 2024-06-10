import argparse

import mlflow

from prefect import flow

import sys
sys.path.append('.')
from load_data import read_data
from transform import transform
from train_LR import train
from config import db_url



MLFLOW_TRACKING_URI = db_url
EXPERIMENT_NAME = "prefect-demo"


@flow
def main(train_path: str, val_path: str) -> None:
    """The main training pipeling"""

    # MLflow setting
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = transform(df_train, df_val)

    # Train the model
    train(X_train, X_val, y_train, y_val, dv)



def parser_opt():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str, default='../data/yellow_tripdata_2023-03.parquet', help="path to train dataset directory.")
    parser.add_argument('--val-path', type=str, default='../data/yellow_tripdata_2023-01.parquet', help="Path to the validation dataset directory.")

    return parser.parse_args()



if __name__ == "__main__":

    args = parser_opt()

    main(args.train_path, args.val_path)
