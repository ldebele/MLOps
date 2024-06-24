
CREATE TABLE IF NOT EXISTS public.evidently_metrics(
    "timestamp" TIMESTAMP,
    "prediction_drift" FLOAT,
    "num_drifted_columns" INT,
    "share_missing_values" FLOAT
)