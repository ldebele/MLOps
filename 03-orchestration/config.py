import os
from sqlalchemy import engine as e


# postgres credentials
DB_HOST = os.environ("DB_HOST")
DB_NAME = os.environ("DB_NAME") 
DB_USER = os.environ("DB_USER")
DB_PASSWORD = os.environ("DB_PASSWORD")
DB_PORT = os.environ("DB_PORT")


def db_url():
    # creating a url 
    return str(e.URL.create(
        drivername="postgresql",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        database=DB_NAME,
        port=DB_PORT
    ))