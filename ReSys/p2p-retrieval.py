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
##  Copurchase products data.
copurchase_products_df = pd.read_sql(
    "SELECT id, query_product_id, candidate_product_id FROM related_product;",
    dbConnection,
)
print(copurchase_products_df.head(5))
copurchase_products = tf.data.Dataset.from_tensor_slices(copurchase_products_df)
## Features of all the available products.
products_df = pd.read_sql("SELECT id, title FROM product;", dbConnection)
print(products_df)
dbConnection.close()
products = tf.data.Dataset.from_tensor_slices(products_df.pop("id"))
## Keep only 'query_product_id' and 'candidate_product_id'
copurchase_products = copurchase_products.map(
    lambda x: {
        "query_product_id": x[1],
        "candidate_product_id": x[2],
    }
)
## For products (show first 5)
print("Products sample:")
for movie in products.take(5):
    print(movie.numpy().decode("utf-8"))
## For copurchase products (show first 5)
print("\nCopurchase products sample:")
for copurchase_product in copurchase_products.take(5):
    print({k: v.numpy().decode("utf-8") for k, v in copurchase_product.items()})
## Let's use a random split, putting 80% of the copurchase products in the train set, and 20% in the test set.
num_copurchase_products = tf.data.experimental.cardinality(copurchase_products).numpy()
train_size = int(num_copurchase_products * 0.8)
test_size = num_copurchase_products - train_size
tf.random.set_seed(42)
shuffled = copurchase_products.shuffle(
    num_copurchase_products, seed=42, reshuffle_each_iteration=False
)
train = shuffled.take(train_size)
test = shuffled.skip(train_size).take(test_size)
## Next we figure out unique query product ids and candidate product ids present in the data so that we can create the embedding query product and candidate product embedding tables.
num_products = tf.data.experimental.cardinality(products).numpy()
product_ids = products.batch(num_products)
query_product_ids = copurchase_products.batch(num_copurchase_products).map(
    lambda x: x["query_product_id"]
)
unique_candidate_product_ids = np.unique(np.concatenate(list(product_ids)))
unique_query_product_ids = np.unique(np.concatenate(list(query_product_ids)))

# Implementing a retrieval model
## We are going to building a two-tower retrieval model by building each tower separately and then combining them in the final model.
## The query tower: The first step is to decide on the dimensionality of the query and candidate representations:
embedding_dimension = 32
## The second is to define the model itself.
query_product_model = tf.keras.Sequential(
    [
        tf.keras.layers.StringLookup(
            vocabulary=unique_query_product_ids, mask_token=None
        ),
        ### We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(
            len(unique_query_product_ids) + 1, embedding_dimension
        ),
    ]
)
## The candidate tower: We can do the same with the candidate tower.
candidate_product_model = tf.keras.Sequential(
    [
        tf.keras.layers.StringLookup(
            vocabulary=unique_candidate_product_ids, mask_token=None
        ),
        tf.keras.layers.Embedding(
            len(unique_candidate_product_ids) + 1, embedding_dimension
        ),
    ]
)
## Metrics: We use the `tfrs.metrics.FactorizedTopK` metric for our retrieval model.
metrics = tfrs.metrics.FactorizedTopK(
    candidates=products.batch(num_products).map(candidate_product_model)
)
## Loss: The next component is the loss used to train our model. We'll make use of the `Retrieval` task object: a convenience wrapper that bundles together the loss function and metric computation:
task = tfrs.tasks.Retrieval(metrics=metrics)


## The full model: We can now put it all together into a model.
class CvosP2PModel(tfrs.Model):

    def __init__(self, query_product_model, candidate_product_model):
        super().__init__()
        self.candidate_product_model: tf.keras.Model = candidate_product_model
        self.query_product_model: tf.keras.Model = query_product_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        ### We pick out the query product features and pass them into the query product model.
        query_product_embeddings = self.query_product_model(
            features["query_product_id"]
        )
        ### And pick out the product features and pass them into the candidate product model,
        ### getting embeddings back.
        positive_product_embeddings = self.candidate_product_model(
            features["candidate_product_id"]
        )
        ### The task computes the loss and the metrics.
        return self.task(query_product_embeddings, positive_product_embeddings)


# Fitting and evaluating
## Let's instantiate the model now.
model = CvosP2PModel(query_product_model, candidate_product_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
## Then shuffle, batch, and cache the training and evaluation data.
cached_train = train.shuffle(num_copurchase_products).batch(8).cache()
cached_test = test.batch(4).cache()
## Then train the  model:
model.fit(cached_train, epochs=3)
## Finally, we can evaluate our model on the test set:
model.evaluate(cached_test, return_dict=True)

# Exporting the model:
RETRIEVAL_QUERY_MODEL_PATH = os.path.join(
    os.getcwd(), "ReSys/models/retrieval-p2p-query-model"
)
RETRIEVAL_CAN_MODEL_PATH = os.path.join(
    os.getcwd(), "ReSys/models/retrieval-p2p-candidate-model"
)
tf.saved_model.save(model.query_product_model, RETRIEVAL_QUERY_MODEL_PATH)
tf.saved_model.save(model.candidate_product_model, RETRIEVAL_CAN_MODEL_PATH)
