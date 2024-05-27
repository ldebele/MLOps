import os 
import pickle
import argparse
import pandas as pd


from sklearn.feature_extraction import DictVectorizer


def wrangle(source: str):

    # read the parquet file
    df = pd.read_parquet(source)

    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


def preprocess(data: pd.DataFrame, dv: DictVectorizer, is_train: bool = False):
    data['PU_DO'] = data['PULocationID'] + '_' + data['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = data[categorical + numerical].to_dict(orient='records')

    if is_train:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv 


def run(source: str, dest_path: str = "./data/cleaned", dataset: str = "green"):
    df_train = wrangle(os.path.join(source, f"{dataset}_tripdata_2023-01.parquet"))
    df_val = wrangle(os.path.join(source, f"{dataset}_tripdata_2023-02.parquet"))
    df_test = wrangle(os.path.join(source, f"{dataset}_tripdata_2023-03.parquet"))

    # Extract the target 
    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values 

    # Fit the DictVectorizer
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, is_train=True)
    X_val, _ = preprocess(df_val, dv)
    X_test, _ = preprocess(df_test, dv)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)



def opt_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--source", type=str, help="Location where the raw NYC taxi trip data was saved.")
    parser.add_argument("--dest-path", type=str, help="Location where the resulting files will be saved")

    return parser.parse_args()


if __name__ == "__main__":

    args = opt_parser() 

    run(args.source, args.dest_path)

    