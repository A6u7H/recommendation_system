import os
import traceback
import logging
import configparser
import numpy as np

from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import HashingTF, IDFModel
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.mllib.linalg.distributed import MatrixEntry, CoordinateMatrix

from runner import SparkRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecSysExecuter():
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config_path = os.path.join("configs", 'config.ini')
        self.config.read(self.config_path)

        self.num_recomendation = self.config.getint(
            "INFERENCE",
            "num_recomendation"
        )
        self.adapter = SparkRunner()
        self.sc = self.adapter.get_context()
        self.spark = self.adapter.get_session()

        self._load_models()
        logger.info("Processor is ready")

    def _load_watched(self):
        """
        Load watched movie matrix
        """
        exp_path = self.config["EXPERIMENTS"]["experiments_path"]
        watched_path = os.path.join(
            exp_path,
            self.config["EXPERIMENTS"]["uimatrix_name"]
        )

        logger.info(f'Reading {watched_path}')
        matrix = self.spark.read.parquet(watched_path)
        self.watched = CoordinateMatrix(
            matrix.rdd.map(lambda row: MatrixEntry(*row))
        )

    def _load_tf(self):
        """
        Load TF model
        """
        exp_path = self.config["EXPERIMENTS"]["experiments_path"]
        tf_path = os.path.join(
            exp_path,
            self.config["EXPERIMENTS"]["tf_name"]
        )

        logger.info(f'Reading {tf_path}')
        self.hashingTF = HashingTF.load(tf_path)

    def _load_idf(self):
        """
        Load IDF model
        """
        exp_path = self.config["EXPERIMENTS"]["experiments_path"]
        idf_path = os.path.join(
            exp_path,
            self.config["EXPERIMENTS"]["idf_name"]
        )

        logger.info(f'Reading {idf_path}')
        self.idf = IDFModel.load(idf_path)

    def _load_idf_features(self):
        """
        Load TF-IDF features
        """
        exp_path = self.config["EXPERIMENTS"]["experiments_path"]
        idf_features_path = os.path.join(
            exp_path,
            self.config["EXPERIMENTS"]["idf_features_name"]
        )

        logger.info(f'Reading {idf_features_path}')
        self.idf_features = self.spark.read.load(idf_features_path)

    def _load_models(self):
        """
        Load all models
        """
        logger.info('Loading Matrix of watched movies')
        self._load_watched()

        logger.info('Loading TF model')
        self._load_tf()

        logger.info('Loading IDF model')
        self._load_idf()

        logger.info('Loading IDF features')
        self._load_idf_features()

    def _get_recomendation(
        self,
        ordered_similarity,
        max_count: int = 5
    ) -> list:
        """
        Get recomendation by user similarity
        :param  ordered_similarity:
        :param max_count: number of recomendation movies
        :return :
        """
        logger.info('Calculate movies ranks')
        users_sim_matrix = IndexedRowMatrix(ordered_similarity)
        multpl = users_sim_matrix.toBlockMatrix().transpose().multiply(
            self.watched.toBlockMatrix()
        )
        ranked_movies = multpl.transpose().toIndexedRowMatrix().rows.sortBy(
            lambda row: row.vector.values[0], ascending=False
        )

        result = []
        for i, row in enumerate(ranked_movies.collect()):
            if i >= max_count:
                break
            result.append((row.index, row.vector[0]))
        return result

    def sample(self):
        logger.info('Sample existing user recomendation')
        temp_matrix = IndexedRowMatrix(self.idf_features.rdd.map(
            lambda row: IndexedRow(row["user_id"], Vectors.dense(row["features"]))
        ))
        temp_block = temp_matrix.toBlockMatrix()
        logger.info('Calculate similarities')
        similarities = temp_block.transpose().toIndexedRowMatrix().columnSimilarities()
        user_id = np.random.randint(low=0, high=self.watched.numCols())

        logger.info(f'Random user ID: {user_id}')
        filtered = similarities.entries.filter(
            lambda x: x.i == user_id or x.j == user_id
        )
        sorted_filtered = filtered.sortBy(lambda x: x.value, ascending=False)
        ordered_similarity = sorted_filtered.map(
            lambda x: IndexedRow(x.j if x.i == user_id else x.i, Vectors.dense(x.value))
        )
        recomendations = self._get_recomendation(ordered_similarity)

        logger.info('TOP recomendations for existing user:')
        for movie_id, rank in recomendations:
            logger.info(f'- movie # {movie_id} (rank: {rank})')


if __name__ == "__main__":
    recsys = RecSysExecuter()
    recsys.sample()
