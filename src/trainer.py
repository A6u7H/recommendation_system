import os
import shutil
import traceback
import configparser
import logging

from pyspark.ml.feature import HashingTF, IDF
from pyspark.mllib.linalg.distributed import MatrixEntry, CoordinateMatrix

from src.runner import SparkRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer():
    def __init__(self) -> None:
        self.config = configparser.ConfigParser()
        self.config_path = os.path.join("configs", "config.ini")
        self.config.read(self.config_path)
        logger.info("Trainer is ready")

    def _remove_stored(self, path: str):
        if os.path.exists(path):
            shutil.rmtree(path)

    def _get_user_item_matrix(self, grouped, path: str):
        flatted_group = grouped.flatMapValues(lambda x: x)
        matrix_group = flatted_group.map(
            lambda x: MatrixEntry(x[0], x[1], 1.0)
        )
        matrix = CoordinateMatrix(matrix_group)
        matrix.entries.toDF().write.parquet(path)
        logger.info(f"User-item matrix saved into {path}")

    def _train_tf(self, grouped, path: str):
        df = grouped.toDF(schema=["user_id", "movie_ids"])
        features_count = self.config.getint(
            "MODEL",
            "features_count",
            fallback=10000
        )
        logger.info(f"TFIDF features count: {features_count}")
        hashingTF = HashingTF(
            inputCol="movie_ids",
            outputCol="rawFeatures",
            numFeatures=features_count
        )

        tf_features = hashingTF.transform(df)
        hashingTF.write().overwrite().save(path)
        logger.info(f"TF model saved into {path}")
        return tf_features

    def _save_idf_features(self, idf_features, path: str):
        idf_features.write.format("parquet").save(path, mode="overwrite")
        logger.info(f"IDF features saved into {path}")

    def _train_idf(self, tf_features, idf_path: str, idf_features_path: str):
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idf = idf.fit(tf_features)

        logger.info(f"IDF model type: {type(idf)}")
        idf.write().overwrite().save(idf_path)
        logger.info(f"IDF model saved into {idf_path}")

        idf_features = idf.transform(tf_features)
        self._save_idf_features(
            idf_features,
            idf_features_path
        )

    def train_models(self, input_filename=None):
        adapter = SparkRunner()
        sc = adapter.get_context()
        _ = adapter.get_session()

        exp_path = self.config["EXPERIMENTS"]["experiments_path"]
        uimatrix_path = os.path.join(
            exp_path,
            self.config["EXPERIMENTS"]["uimatrix_name"]
        )
        tf_path = os.path.join(
            exp_path,
            self.config["EXPERIMENTS"]["tf_name"]
        )
        idf_path = os.path.join(
            exp_path,
            self.config["EXPERIMENTS"]["idf_name"]
        )
        idf_features_path = os.path.join(
            exp_path,
            self.config["EXPERIMENTS"]["idf_features_name"]
        )

        self._remove_stored(exp_path)

        input_filename = self.config["DATA"]["input_file"]
        logger.info(f"train data filename: {input_filename}")

        grouped = sc.textFile(input_filename, adapter.num_parts) \
            .map(lambda x: map(int, x.split())).groupByKey() \
            .map(lambda x: (x[0], list(x[1])))

        logger.info("Calculating user-item matrix...")
        self._get_user_item_matrix(
            grouped,
            uimatrix_path
        )

        logger.info("Train TF model")
        tf_features = self._train_tf(
            grouped,
            tf_path
        )

        logger.info("Train IDF model")
        self._train_idf(
            tf_features,
            idf_path,
            idf_features_path
        )


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_models()
