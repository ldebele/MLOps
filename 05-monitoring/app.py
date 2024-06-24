
import os
import pickle

import pandas as pd

import uvicorn
from fastapi import FastAPI
from typing import BaseModel


with open('../models/model.bin', 'rb') as f:
    dv, model = pickle.load(f)

def preprocess(df):
    categorical = ['PULocationID', 'DOLocationID']
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def predict_features(features):
    """" """
    X = dv.transform(features)
    y_pred = model.predict(X)[0]

    return round(y_pred)


def save_to_evidently_service(record, prediction):
    pass 

def save_to_db(ride, prediction):
    pass



app = FastAPI("Monitoring ML Prediction")


class RideData(BaseModel):
    pass


@app.post("/predict")
def predict(data: RideData):
    """"""

    df = pd.read_csv(data.dict())
    ride = preprocess(df)

    pred = predict_features(ride)
    print("Finished prediction")

    return_dict = {"duration_init": pred}

    save_to_db(ride, pred)
    save_to_evidently_service(ride, pred)

    return return_dict


if __name__=="__main__":
    
    uvicorn.run()

