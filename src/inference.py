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

logger = logging.getLogger(__name__)


class Processor():
    def __init__(self):
        """
        default initialization
        """
        self.config = configparser.ConfigParser()
        self.config_path = os.path.join("configs", 'config.ini')
        self.config.read(self.config_path)
        try:
            self.adapter = SparkRunner()
            self.sc = self.adapter.get_context()
            self.spark = self.adapter.get_session()
        except:
            logger.error(traceback.format_exc())

        if not self._load_models():
            raise Exception('Can\'t load models')
        logger.info("Processor is ready")

    def _load_watched(self) -> bool:
        """
        Load watched movie matrix
        """
        path = self.config.get("MODEL", "WATCHED_PATH")
        if path is None or not os.path.exists(path):
            logger.error('Matrix of watched movies doesn\'t exists')
            return False

        logger.info(f'Reading {path}')
        try:
            matrix = self.spark.read.parquet(path)
            self.watched = CoordinateMatrix(
                matrix.rdd.map(lambda row: MatrixEntry(*row))
            )
        except:
            logger.error(traceback.format_exc())
            return False
        return True

    def _load_tf(self) -> bool:
        """
        Load TF model
        """
        path = self.config.get("MODEL", "TF_PATH")
        if path is None or not os.path.exists(path):
            logger.error('TF model doesn\'t exists')
            return False

        logger.info(f'Reading {path}')
        try:
            self.hashingTF = HashingTF.load(path)
        except:
            logger.error(traceback.format_exc())
            return False
        return True

    def _load_idf(self) -> bool:
        """
        Load IDF model
        """
        path = self.config.get("MODEL", "IDF_PATH")
        if path is None or not os.path.exists(path):
            logger.error('IDF model doesn\'t exists')
            return False

        logger.info(f'Reading {path}')
        try:
            self.idf = IDFModel.load(path)
        except:
            logger.error(traceback.format_exc())
            return False
        return True

    def _load_idf_features(self) -> bool:
        """
        Load TF-IDF features
        """
        path = self.config.get("MODEL", "IDF_FEATURES_PATH")
        if path is None or not os.path.exists(path):
            logger.error('IDF features doesn\'t exists')
            return False

        logger.info(f'Reading {path}')
        try:
            self.idf_features = self.spark.read.load(path)
        except:
            logger.error(traceback.format_exc())
            return False
        return True

    def _load_models(self) -> bool:
        """
        Load all models
        """
        logger.info('Loading Matrix of watched movies')
        if not self._load_watched():
            return False

        logger.info('Loading TF model')
        if not self._load_tf():
            return False

        logger.info('Loading IDF model')
        if not self._load_idf():
            return False

        logger.info('Loading IDF features')
        if not self._load_idf_features():
            return False

        return True

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
        multpl = users_sim_matrix.toBlockMatrix().transpose().multiply(self.watched.toBlockMatrix())
        ranked_movies = multpl.transpose().toIndexedRowMatrix().rows.sortBy(lambda row: row.vector.values[0], ascending=False)

        result = []
        for i, row in enumerate(ranked_movies.collect()):
            if i >= max_count:
                break
            result.append((row.index, row.vector[0]))
        return result

    def sample(self):
        """
        Выводит рекомендации для случайно выбранного пользователя из датасета
        """
        logger.info('Sample existing user recomendation')
        temp_matrix = IndexedRowMatrix(self.idf_features.rdd.map(
            lambda row: IndexedRow(row["user_id"], Vectors.dense(row["features"]))
        ))
        temp_block = temp_matrix.toBlockMatrix()
        logger.info('Calculate similarities')
        similarities = temp_block.transpose().toIndexedRowMatrix().columnSimilarities()
        user_id = np.random.randint(low=0, high=self.watched.numCols())
        logger.info(f'Random user ID: {user_id}')
        filtered = similarities.entries.filter(lambda x: x.i == user_id or x.j == user_id)
        ordered_similarity = filtered.sortBy(lambda x: x.value, ascending=False) \
            .map(lambda x: IndexedRow(x.j if x.i == user_id else x.i, Vectors.dense(x.value)))

        recomendations = self._get_recomendation(ordered_similarity)
        logger.info('TOP recomendations for existing user:')
        for movie_id, rank in recomendations:
            logger.info(f'- movie # {movie_id} (rank: {rank})')


if __name__ == "__main__":
    processor = Processor()
    processor.sample()
