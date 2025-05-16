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
    "SELECT id, query_product_id, candidate_product_id, copurchase_frequency FROM related_product;",
    dbConnection,
)
copurchase_products_df["copurchase_frequency"] = copurchase_products_df[
    "copurchase_frequency"
].astype(np.float32)
print(copurchase_products_df.head(5))
copurchase_products = tf.data.Dataset.from_tensor_slices(
    {
        "query_product_id": copurchase_products_df["query_product_id"].values,
        "candidate_product_id": copurchase_products_df["candidate_product_id"].values,
        "copurchase_frequency": copurchase_products_df["copurchase_frequency"].values,
    }
)
## For copurchase products (show first 5)
print("\nCopurchase products sample:")
for copurchase_product in copurchase_products.take(5):
    print({k: v for k, v in copurchase_product.items()})

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
candidate_product_ids = copurchase_products.batch(num_copurchase_products).map(
    lambda x: x["candidate_product_id"]
)
query_product_ids = copurchase_products.batch(num_copurchase_products).map(
    lambda x: x["query_product_id"]
)
unique_candidate_product_ids = np.unique(np.concatenate(list(candidate_product_ids)))
unique_query_product_ids = np.unique(np.concatenate(list(query_product_ids)))


# Implementing a ranking model
## Architecture:
## Ranking models do not face the same efficiency constraints as retrieval models do, and so we have a little bit more freedom in our choice of architectures. We can implement our ranking model as follows:
class RankingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        embedding_dimension = 32

        ### Compute embeddings for query products.
        self.query_product_embeddings = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=unique_query_product_ids, mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(unique_query_product_ids) + 1, embedding_dimension
                ),
            ]
        )

        ### Compute embeddings for candidate products.
        self.candidate_product_embeddings = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=unique_candidate_product_ids, mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(unique_candidate_product_ids) + 1, embedding_dimension
                ),
            ]
        )

        ### Compute predictions.
        self.copurchase_frequency = tf.keras.Sequential(
            [
                ### Learn multiple dense layers.
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                ### Make copurchase frequency predictions in the final layer.
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, inputs):

        query_product_id, candidate_product_id = inputs

        query_product_embedding = self.query_product_embeddings(query_product_id)
        candidate_product_embedding = self.candidate_product_embeddings(
            candidate_product_id
        )

        return self.copurchase_frequency(
            tf.concat([query_product_embedding, candidate_product_embedding], axis=1)
        )


## Loss and metrics:
## We'll make use of the `Ranking` task object: a convenience wrapper that bundles together the loss function and metric computation.
## We'll use it together with the `MeanSquaredError` Keras loss in order to predict the copurchase_frequency.
task = tfrs.tasks.Ranking(
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
)


class CvosP2PModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model(
            (features["query_product_id"], features["candidate_product_id"])
        )

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        labels = features.pop("copurchase_frequency")

        copurchase_frequency_predictions = self(features)

        ### The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=copurchase_frequency_predictions)


# Fitting and Evaluating
## After defining the model, we can use standard Keras fitting and evaluation routines to fit and evaluate the model.
## Let's first instantiate the model.
model = CvosP2PModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
## Then shuffle, batch, and cache the training and evaluation data.
cached_train = train.shuffle(num_copurchase_products).batch(8).cache()
cached_test = test.batch(4).cache()
## Then train the model:
model.fit(cached_train, epochs=3)
## As the model trains, the loss is falling and the RMSE metric is improving.
## Finally, we can evaluate our model on the test set:
model.evaluate(cached_test, return_dict=True)
## The lower the RMSE metric, the more accurate our model is at predicting copurchase frequency.

# Exporting the model:
RANKING_MODEL_PATH = os.path.join(os.getcwd(), "ReSys/models/ranking-p2p-model")
tf.saved_model.save(model, RANKING_MODEL_PATH)
