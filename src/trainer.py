import os
import shutil
import traceback
import configparser
import logging

from pyspark.ml.feature import HashingTF, IDF
from pyspark.mllib.linalg.distributed import MatrixEntry, CoordinateMatrix

from runner import SparkRunner

logger = logging.getLogger(__name__)


class Trainer():
    def __init__(self) -> None:
        self.config = configparser.ConfigParser()
        self.config_path = os.path.join("configs", "config.ini")
        self.config.read(self.config_path)
        logger.info("Trainer is ready")

    def _remove_stored(self, path) -> bool:
        """
        Удаление сохраненной ранее модели
        """
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        if os.path.exists(path):
            logger.error(f"Can\'t remove {path}")
            return False
        return True

    def _calc_watched_matrix(self, grouped, path="./models/WATCHED") -> bool:
        """
        Подготовка матрицы просмотренных фильмов
        """
        if not self._remove_stored(path):
            return False

        flatted_group = grouped.flatMapValues(lambda x: x)
        matrix_group = flatted_group.map(lambda x: MatrixEntry(x[0], x[1], 1.0))
        matrix = CoordinateMatrix(matrix_group)
        try:
            matrix.entries.toDF().write.parquet(path)
            self.config.set('MODEL', 'WATCHED_PATH', path)
            logger.info(f"Matrix of watched movies is stored at {path}")
        except:
            logger.error(traceback.format_exc())
            return False

        return os.path.exists(path)

    def _train_tf(self, grouped, path="./models/TF_MODEL"):
        """
        Обучение TF модели
        """
        if not self._remove_stored(path):
            return None

        df = grouped.toDF(schema=["user_id", "movie_ids"])
        FEATURES_COUNT = self.config.getint(
            "MODEL",
            "FEATURES_COUNT",
            fallback=10000
        )
        logger.info(f"TF-IDF features count = {FEATURES_COUNT}")
        hashingTF = HashingTF(
            inputCol="movie_ids",
            outputCol="rawFeatures",
            numFeatures=FEATURES_COUNT
        )

        tf_features = hashingTF.transform(df)
        try:
            hashingTF.write().overwrite().save(path)
            self.config.set("MODEL", "TF_PATH", path)
            logger.info(f"TF model stored at {path}")
        except:
            logger.error(traceback.format_exc())
            return None

        return tf_features

    def _save_idf_features(self, idf_features, path="./models/IDF_FEATURES") -> bool:
        if not self._remove_stored(path):
            return False
        try:
            idf_features.write.format("parquet").save(path, mode="overwrite")
            self.config["MODEL"]["IDF_FEATURES_PATH"] = path
            logger.info(f"IDF features stored at {path}")
        except:
            logger.error(traceback.format_exc())
            return False
        return True

    def _train_idf(self, tf_features, path="./models/IDF_MODEL") -> bool:
        """
        Create IDF features
        """
        if not self._remove_stored(path):
            return False
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idf = idf.fit(tf_features)

        logger.info(f"IDF model type: {type(idf)}")
        try:
            idf.write().overwrite().save(path)
            self.config.set("MODEL", "IDF_PATH", path)
            logger.info(f"IDF model stored at {path}")
        except:
            logger.error(traceback.format_exc())
            return False

        idf_features = idf.transform(tf_features)
        if not self._save_idf_features(idf_features):
            return False
        return True

    def train_models(self, input_filename=None) -> bool:
        try:
            adapter = SparkRunner()
            sc = adapter.get_context()
            _ = adapter.get_session()
        except:
            logger.error(traceback.format_exc())
            return False

        if input_filename is None:
            INPUT_FILENAME = self.config.get(
                "DATA",
                "INPUT_FILE",
                fallback="./data/generated.csv"
            )
        else:
            INPUT_FILENAME = input_filename
        logger.info(f"train data filename = {INPUT_FILENAME}")
        grouped = sc.textFile(INPUT_FILENAME, adapter.num_parts) \
            .map(lambda x: map(int, x.split())).groupByKey() \
            .map(lambda x: (x[0], list(x[1])))

        logger.info("Calculating matrix of watched movies")
        if not self._calc_watched_matrix(grouped):
            return False

        logger.info("Train TF model")
        tf_features = self._train_tf(grouped)
        if tf_features is None:
            return False

        logger.info("Train IDF model")
        if not self._train_idf(tf_features):
            return False

        os.remove(self.config_path)
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)

        return True


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_models()
