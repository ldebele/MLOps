#!/usr/bin/env python

import pickle
import argparse
from datetime import datetime

import pandas as pd



with open('/app/model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)



def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



def run(year: int, month: int, output_file = './prediction.parquet'):


    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    print(url)

    df = read_data(url)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    start_time = datetime.now()
    y_pred = model.predict(X_val)
    end_time = datetime.now()

    time_difference = end_time - start_time
    total_time = time_difference.total_seconds()
    print(f"Total prediction time: {total_time}")

    df_result = pd.DataFrame()
    df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result['y_pred'] = y_pred
    
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--year", type=int, help="year of the dataset")
    parser.add_argument("--month", type=int, help="Months of the dataset")
    args = parser.parse_args()

    categorical = ['PULocationID', 'DOLocationID']

    run(args.year, args.month)