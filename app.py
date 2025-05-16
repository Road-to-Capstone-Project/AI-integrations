from flask import Flask
from flask_cors import CORS
from flask import jsonify, make_response, request
import json, requests
import numpy as np
import os
import tensorflow as tf
import tensorflow_recommenders as tfrs
import os
from tensorflow import keras
from dotenv import load_dotenv
import pandas as pd

# import psycopg2
from sqlalchemy import create_engine
import subprocess


RETRIEVAL_QUERY_MODEL_PATH = os.path.join(
    os.getcwd(), "ReSys/models/retrieval-c2p-query-model"
)
RETRIEVAL_CAN_MODEL_PATH = os.path.join(
    os.getcwd(), "ReSys/models/retrieval-c2p-candidate-model"
)
RANKING_MODEL_PATH = os.path.join(os.getcwd(), "ReSys/models/ranking-c2p-model")
c2p_ranking_model = tf.saved_model.load(RANKING_MODEL_PATH)
c2p_query_model = tf.saved_model.load(RETRIEVAL_QUERY_MODEL_PATH)
c2p_candidate_model = tf.saved_model.load(RETRIEVAL_CAN_MODEL_PATH)
RETRIEVAL_QUERY_MODEL_PATH = os.path.join(
    os.getcwd(), "ReSys/models/retrieval-p2p-query-model"
)
RETRIEVAL_CAN_MODEL_PATH = os.path.join(
    os.getcwd(), "ReSys/models/retrieval-p2p-candidate-model"
)
RANKING_MODEL_PATH = os.path.join(os.getcwd(), "ReSys/models/ranking-p2p-model")
p2p_ranking_model = tf.saved_model.load(RANKING_MODEL_PATH)
p2p_query_model = tf.saved_model.load(RETRIEVAL_QUERY_MODEL_PATH)
p2p_candidate_model = tf.saved_model.load(RETRIEVAL_CAN_MODEL_PATH)

## Set up the database connection
load_dotenv()
endpoint = os.getenv("DATABASE_URL")
alchemyEngine = create_engine(endpoint)


app = Flask(__name__)
CORS(app)


@app.route("/c2p-recommend", methods=["GET"])
def get_c2p_recommendations():
    try:
        customer_id = request.args.get("customer_id")
        if not customer_id:
            return make_response(jsonify({"error": "customer_id is required"}), 400)
        NUM_OF_CANDIDATES = int(request.args.get("limit"))
        if not NUM_OF_CANDIDATES:
            return make_response(jsonify({"error": "limit is required"}), 400)

        dbConnection = alchemyEngine.connect()
        ## Features of all the available products.
        products_df = pd.read_sql("SELECT id, title FROM product;", dbConnection)
        print(products_df)
        dbConnection.close()
        products = tf.data.Dataset.from_tensor_slices(products_df.pop("id"))
        ## Create a BruteForce index using the SavedModel query model
        index = tfrs.layers.factorized_top_k.BruteForce(c2p_query_model)
        ## Use the SavedModel candidate model to map products to embeddings
        index.index_from_dataset(
            tf.data.Dataset.zip(
                (products.batch(2), products.batch(2).map(c2p_candidate_model))
            )
        )
        ## Get recommendations
        _, titles = index(tf.constant([customer_id]))

        product_candidates = [
            t.decode("utf-8") for t in titles[0, :NUM_OF_CANDIDATES].numpy()
        ]
        products_scores = []
        for product_id in product_candidates:
            products_scores.append(
                c2p_ranking_model(
                    {"customer_id": np.array([customer_id]), "product_id": [product_id]}
                ).numpy()[0][0]
            )
        ranked_products = [
            m[1]
            for m in sorted(
                list(zip(products_scores, product_candidates)), reverse=True
            )
        ]
        return make_response(jsonify({"product_ids": ranked_products}), 200)
    except Exception as e:
        return make_response(
            jsonify({"error": f"Recommending failed because {e}"}), 500
        )


@app.route("/p2p-recommend", methods=["GET"])
def get_p2p_recommendations():
    try:
        query_product_id = request.args.get("product_id")
        if not query_product_id:
            return make_response(jsonify({"error": "product_id is required"}), 400)
        NUM_OF_CANDIDATES = int(request.args.get("limit"))
        if not NUM_OF_CANDIDATES:
            return make_response(jsonify({"error": "limit is required"}), 400)

        dbConnection = alchemyEngine.connect()
        ## Features of all the available products.
        products_df = pd.read_sql("SELECT id, title FROM product;", dbConnection)
        print(products_df)
        dbConnection.close()
        products = tf.data.Dataset.from_tensor_slices(products_df.pop("id"))
        ## Create a BruteForce index using the SavedModel query model
        index = tfrs.layers.factorized_top_k.BruteForce(p2p_query_model)
        ## Use the SavedModel candidate model to map products to embeddings
        index.index_from_dataset(
            tf.data.Dataset.zip(
                (products.batch(2), products.batch(2).map(p2p_candidate_model))
            )
        )
        ## Get recommendations
        _, titles = index(tf.constant([query_product_id]))

        product_candidates = [
            t.decode("utf-8") for t in titles[0, :NUM_OF_CANDIDATES].numpy()
        ]
        products_scores = []
        for product_id in product_candidates:
            products_scores.append(
                p2p_ranking_model(
                    {
                        "query_product_id": np.array([query_product_id]),
                        "candidate_product_id": [product_id],
                    }
                ).numpy()[0][0]
            )
        ranked_products = [
            m[1]
            for m in sorted(
                list(zip(products_scores, product_candidates)), reverse=True
            )
        ]
        return make_response(jsonify({"product_ids": ranked_products}), 200)
    except Exception as e:
        return make_response(
            jsonify({"error": f"Recommending failed because {e}"}), 500
        )


@app.route("/train-recommendation-models", methods=["GET"])
def train_recommendation_models():
    try:
        print("Training models...")
        retrieval_args = ["python", "./ReSys/c2p-retrieval.py"]
        subprocess.run(retrieval_args)
        ranking_args = ["python", "./ReSys/c2p-ranking.py"]
        subprocess.run(ranking_args)
        print("Training models...")
        retrieval_args = ["python", "./ReSys/p2p-retrieval.py"]
        subprocess.run(retrieval_args)
        ranking_args = ["python", "./ReSys/p2p-ranking.py"]
        subprocess.run(ranking_args)
        print("Training completed successfully.")
        return "", 200
    except Exception as e:
        return make_response(jsonify({"error": f"Training failed because {e}"}), 500)


if __name__ == "__main__":
    app.run(debug=True)
