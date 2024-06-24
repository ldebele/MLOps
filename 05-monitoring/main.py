import io
import uuid
import pytz
import time
import joblib
import psycopg2
import logging
import datetime

import pandas as pd

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


SEND_TIMEOUT = 10

reference_data = pd.read_parquet('./data/reference.parquet')
with open('./models/lin_reg.bin', 'rb') as f:
    model = joblib.load(f)

raw_data = pd.read_parquet('./data/green_tripdata_2023-02.parquet')

begin = datetime.datetime(2023, 2, 1, 0, 0)
num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
cat_features = ['PULocationID', 'DOLocationID']

column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

report = Report(metrics = [ColumnDriftMetric(column_name='prediction'),
                           DatasetDriftMetric(),
                           DatasetMissingValuesMetric()]
                )


def calculate_metrics(curr, i):
    current_data = raw_data[(raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i))) &
		(raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))]
    
    current_data['prediction'] = model.predict(current_data[num_features + cat_features].fillna(0))

    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

    # insert the data into database.
    curr.execute(
		"insert into evidently_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
		(begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_values)
	)


def create_db():
    host="localhost"
    db_name="grafana_db"
    user="grafana_user"
    password="user123"
    port=5432

    conn = psycopg2.connect(
        host=host,
        database=db_name,
        user=user,
        password=password,
        port=port
    )

    cursor = conn.cursor()

    with open('./schema.sql', 'r') as f:
        schema= f.read()
        cursor.execute(schema)
        conn.commit()  
        logging.info("Database Table is successfully created.")

    cursor.close()
    conn.close()


def batch_monitoring_backfill():
    create_db()
    host="localhost"
    db_name="grafana_db"
    user="grafana_user"
    password="user123"
    port=5432


    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg2.connect(host=host, database=db_name, user=user, password=password, port=port) as conn:
        for i in range(0, 27):

            with conn.cursor() as curr:
                calculate_metrics(curr, i)
                
            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
                
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
                
            logging.info("data sent")




if __name__ == "__main__":
    batch_monitoring_backfill()

    


            




