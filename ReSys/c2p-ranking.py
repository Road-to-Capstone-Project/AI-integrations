import datetime
import os
from dotenv import load_dotenv
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

# Preparing the dataset
## Set up the database connection
load_dotenv()
endpoint = os.getenv('DATABASE_URL')
alchemyEngine = create_engine(endpoint)
dbConnection = alchemyEngine.connect()
##  Ratings data.
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

## Let's use a random split, putting 80% of the ratings in the train set, and 20% in the test set.
num_ratings = tf.data.experimental.cardinality(ratings).numpy()
train_size = int(num_ratings * 0.8)
test_size = num_ratings - train_size
tf.random.set_seed(42)
shuffled = ratings.shuffle(num_ratings, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(train_size)
test = shuffled.skip(train_size).take(test_size)
## Next we figure out unique customer ids and product ids present in the data so that we can create the embedding customer and product embedding tables.
product_ids = ratings.batch(num_ratings).map(lambda x: x["product_id"])
customer_ids = ratings.batch(num_ratings).map(lambda x: x["customer_id"])
unique_product_titles = np.unique(np.concatenate(list(product_ids)))
unique_customer_ids = np.unique(np.concatenate(list(customer_ids)))

# Implementing a ranking model
## Architecture:
## Ranking models do not face the same efficiency constraints as retrieval models do, and so we have a little bit more freedom in our choice of architectures. We can implement our ranking model as follows:
class RankingModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    embedding_dimension = 32

    ### Compute embeddings for customers.
    self.customer_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_customer_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_customer_ids) + 1, embedding_dimension)
    ])

    ### Compute embeddings for products.
    self.product_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_product_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_product_titles) + 1, embedding_dimension)
    ])

    ### Compute predictions.
    self.ratings = tf.keras.Sequential([
      ### Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      ### Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
  ])
    
  def call(self, inputs):

    customer_id, product_id = inputs

    customer_embedding = self.customer_embeddings(customer_id)
    product_embedding = self.product_embeddings(product_id)

    return self.ratings(tf.concat([customer_embedding, product_embedding], axis=1))

## Loss and metrics:
## We'll make use of the `Ranking` task object: a convenience wrapper that bundles together the loss function and metric computation. 
## We'll use it together with the `MeanSquaredError` Keras loss in order to predict the ratings.
task = tfrs.tasks.Ranking(
  loss = tf.keras.losses.MeanSquaredError(),
  metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
class CvosC2PModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel()
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model(
        (features["customer_id"], features["product_id"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("rating")
    
    rating_predictions = self(features)

    ### The task computes the loss and the metrics.
    return self.task(labels=labels, predictions=rating_predictions)

# Fitting and Evaluating
## After defining the model, we can use standard Keras fitting and evaluation routines to fit and evaluate the model.
## Let's first instantiate the model.
model = CvosC2PModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
## Then shuffle, batch, and cache the training and evaluation data.
cached_train = train.shuffle(num_ratings).batch(8192).cache()
cached_test = test.batch(4096).cache()
## Then train the  model:
model.fit(cached_train, epochs=3)
## As the model trains, the loss is falling and the RMSE metric is improving.
## Finally, we can evaluate our model on the test set:
model.evaluate(cached_test, return_dict=True)
## The lower the RMSE metric, the more accurate our model is at predicting ratings.

# Exporting the model:
RANKING_MODEL_PATH = os.path.join(os.getcwd(), "ReSys/models/ranking-c2p-model")
tf.saved_model.save(model, RANKING_MODEL_PATH)
