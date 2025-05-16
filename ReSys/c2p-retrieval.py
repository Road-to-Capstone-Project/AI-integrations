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
endpoint = os.getenv("DATABASE_URL")
alchemyEngine = create_engine(endpoint)
dbConnection = alchemyEngine.connect()
##  Ratings data.
ratings_df = pd.read_sql(
    "SELECT id, customer_id, product_id FROM review;", dbConnection
)
print(ratings_df.head(5))
ratings = tf.data.Dataset.from_tensor_slices(ratings_df)
## Features of all the available products.
products_df = pd.read_sql("SELECT id, title FROM product;", dbConnection)
print(products_df)
dbConnection.close()
products = tf.data.Dataset.from_tensor_slices(products_df.pop("id"))
## Keep only 'product_id' and 'customer_id'
ratings = ratings.map(
    lambda x: {
        "product_id": x[1],
        "customer_id": x[2],
    }
)
## For products (show first 5)
print("Products sample:")
for movie in products.take(5):
    print(movie.numpy().decode("utf-8"))
## For ratings (show first 5)
print("\nRatings sample:")
for rating in ratings.take(5):
    print({k: v.numpy().decode("utf-8") for k, v in rating.items()})
## Let's use a random split, putting 80% of the ratings in the train set, and 20% in the test set.
num_ratings = tf.data.experimental.cardinality(ratings).numpy()
train_size = int(num_ratings * 0.8)
test_size = num_ratings - train_size
tf.random.set_seed(42)
shuffled = ratings.shuffle(num_ratings, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(train_size)
test = shuffled.skip(train_size).take(test_size)
## Next we figure out unique user ids and product ids present in the data so that we can create the embedding user and product embedding tables.
num_products = tf.data.experimental.cardinality(products).numpy()
product_ids = products.batch(num_products)
customer_ids = ratings.batch(num_ratings).map(lambda x: x["customer_id"])
unique_product_ids = np.unique(np.concatenate(list(product_ids)))
unique_customer_ids = np.unique(np.concatenate(list(customer_ids)))

# Implementing a retrieval model
## We are going to building a two-tower retrieval model by building each tower separately and then combining them in the final model.
## The query tower: The first step is to decide on the dimensionality of the query and candidate representations:
embedding_dimension = 32
## The second is to define the model itself.
customer_model = tf.keras.Sequential(
    [
        tf.keras.layers.StringLookup(vocabulary=unique_customer_ids, mask_token=None),
        ### We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(len(unique_customer_ids) + 1, embedding_dimension),
    ]
)
## The candidate tower: We can do the same with the candidate tower.
product_model = tf.keras.Sequential(
    [
        tf.keras.layers.StringLookup(vocabulary=unique_product_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_product_ids) + 1, embedding_dimension),
    ]
)
## Metrics: We use the `tfrs.metrics.FactorizedTopK` metric for our retrieval model.
metrics = tfrs.metrics.FactorizedTopK(
    candidates=products.batch(num_products).map(product_model)
)
## Loss: The next component is the loss used to train our model. We'll make use of the `Retrieval` task object: a convenience wrapper that bundles together the loss function and metric computation:
task = tfrs.tasks.Retrieval(metrics=metrics)


## The full model: We can now put it all together into a model.
class CvosC2PModel(tfrs.Model):

    def __init__(self, customer_model, product_model):
        super().__init__()
        self.product_model: tf.keras.Model = product_model
        self.customer_model: tf.keras.Model = customer_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        ### We pick out the customer features and pass them into the customer model.
        customer_embeddings = self.customer_model(features["customer_id"])
        ### And pick out the product features and pass them into the product model,
        ### getting embeddings back.
        positive_product_embeddings = self.product_model(features["product_id"])
        ### The task computes the loss and the metrics.
        return self.task(customer_embeddings, positive_product_embeddings)


# Fitting and evaluating
## Let's instantiate the model now.
model = CvosC2PModel(customer_model, product_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
## Then shuffle, batch, and cache the training and evaluation data.
cached_train = train.shuffle(num_ratings).batch(8192).cache()
cached_test = test.batch(4096).cache()
## Then train the  model:
model.fit(cached_train, epochs=3)
## Finally, we can evaluate our model on the test set:
model.evaluate(cached_test, return_dict=True)

# Exporting the model:
RETRIEVAL_QUERY_MODEL_PATH = os.path.join(
    os.getcwd(), "ReSys/models/retrieval-c2p-query-model"
)
RETRIEVAL_CAN_MODEL_PATH = os.path.join(
    os.getcwd(), "ReSys/models/retrieval-c2p-candidate-model"
)
tf.saved_model.save(model.customer_model, RETRIEVAL_QUERY_MODEL_PATH)
tf.saved_model.save(model.product_model, RETRIEVAL_CAN_MODEL_PATH)
