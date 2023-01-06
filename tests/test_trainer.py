import os

from src.trainer import Trainer

trainer = Trainer()

def test_train_models():
    trainer.train_models("./data/generated.csv")
    assert os.path.exists("experiments/idf_features")
    assert os.path.exists("experiments/idf_model")
    assert os.path.exists("experiments/tf_model")
    assert os.path.exists("experiments/watched_matrix")
