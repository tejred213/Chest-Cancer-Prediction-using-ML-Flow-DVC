# import tensorflow as tf
# from pathlib import Path
# import mlflow
# import mlflow.keras
# from urllib.parse import urlparse
# from cnnClassifier.entity.config_entity import EvaluationConfig
# from cnnClassifier.utils.common import read_yaml, create_directories, save_json

# tf.keras.__version__ = tf.__version__

# class Evaluation:
#     def __init__(self, config: EvaluationConfig):
#         self.config = config

#     def _valid_generator(self):
#         datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.10)

#         dataflow_kwargs = dict(
#             target_size=self.config.params_image_size[:-1],
#             batch_size=self.config.params_batch_size,
#             interpolation="bilinear",
#         )

#         valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#             **datagenerator_kwargs
#         )

#         self.valid_generator = valid_datagenerator.flow_from_directory(
#             directory=self.config.training_data,
#             subset="validation",
#             shuffle=False,
#             class_mode="categorical",
#             **dataflow_kwargs,
#         )

#     @staticmethod
#     def load_model(path: Path) -> tf.keras.Model:
#         return tf.keras.models.load_model(path)

#     def evaluation(self):
#         self.model = self.load_model(self.config.path_of_model)
#         self._valid_generator()
#         self.score = self.model.evaluate(self.valid_generator)
#         self.save_score()

#     def save_score(self):
#         scores = {"loss": self.score[0], "accuracy": self.score[1]}
#         save_json(path=Path("scores.json"), data=scores)

#     def log_into_mlflow(self):
#         mlflow.set_registry_uri(self.config.mlflow_uri)
#         tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
#         print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
#         print(f"Tracking URL Type: {tracking_url_type_store}")

#         with mlflow.start_run() as run:
#             print(f"Active Run ID: {run.info.run_id}")
#             try:
#                 mlflow.log_params(self.config.all_params)
#                 print("Logged params:", self.config.all_params)
#                 mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
#                 print("Logged metrics:", {"loss": self.score[0], "accuracy": self.score[1]})

#                 if tracking_url_type_store != "file":
#                     mlflow.keras.log_model(
#                         self.model, "model", registered_model_name="VGG16Model"
#                     )
#                     print("Model registered with name: VGG16Model")
#                 else:
#                     mlflow.keras.log_model(self.model, "model")
#                     print("Model logged locally.")
#             except Exception as e:
#                 print("MLflow logging failed:", str(e))



import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json

# This line is not needed or is a typo: tf.keras._version_ = tf._version_
# Possibly you intended to print them, e.g.:
print("TF Version:", tf.__version__)

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.valid_generator = None

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.10
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode='categorical'
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Print class indices here to see how the classes are mapped
        print("Evaluation class indices:", self.valid_generator.class_indices)

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
        print("Evaluation scores saved:", scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Tracking URL Type: {tracking_url_type_store}")

        with mlflow.start_run() as run:
            print(f"Active Run ID: {run.info.run_id}")
            try:
                mlflow.log_params(self.config.all_params)
                print("Logged params:", self.config.all_params)

                mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
                print("Logged metrics:", {"loss": self.score[0], "accuracy": self.score[1]})

                if tracking_url_type_store != "file":
                    mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
                    print("Model registered with name: VGG16Model")
                else:
                    mlflow.keras.log_model(self.model, "model")
                    print("Model logged locally.")
            except Exception as e:
                print("MLflow logging failed:", str(e))