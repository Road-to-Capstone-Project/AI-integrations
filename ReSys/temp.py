import os
from dotenv import load_dotenv
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

load_dotenv()
endpoint = os.getenv('DATABASE_URL')
alchemyEngine = create_engine(endpoint)
dbConnection = alchemyEngine.connect()
ratings_df = pd.read_sql("SELECT id, customer_id, product_id, rating FROM review;", dbConnection)
ratings_df["rating"] = ratings_df["rating"].astype(np.float32)  
print(ratings_df.head(5))
ratings = tf.data.Dataset.from_tensor_slices({
    "customer_id": ratings_df["customer_id"].values,
    "product_id": ratings_df["product_id"].values,
    "rating": ratings_df["rating"].values,
})

## For ratings (show first 5)
print("\nRatings sample:")
for rating in ratings.take(5):
    print({k: v for k, v in rating.items()})

ratings = tfds.load("movielens/100k-ratings", split="train")
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})
## For ratings (show first 5)
print("\nRatings sample:")
for rating in ratings.take(5):
    print({k: v for k, v in rating.items()})
