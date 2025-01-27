# import os
# import tensorflow as tf
# from pathlib import Path
# from cnnClassifier.entity.config_entity import TrainingConfig

# class Training:
#     def __init__(self, config: TrainingConfig):
#         """
#         Initializes the Training class with the provided configuration settings.
#         """
#         self.config = config
#         self.model = None
#         self.train_generator = None
#         self.valid_generator = None

#     def get_base_model(self):
#         """
#         Load the base model from the configured path.
#         """
#         try:
#             self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
#             print(f"Base model loaded from {self.config.updated_base_model_path}")
#         except Exception as e:
#             raise ValueError(f"Error loading model from {self.config.updated_base_model_path}: {e}")

#     def train_valid_generator(self):
#         """
#         Set up the training and validation data generators, ensuring the training dataset
#         is significantly larger than the validation dataset.
#         """
#         if not self.config.training_data.exists():
#             raise FileNotFoundError(f"Training data directory does not exist: {self.config.training_data}")

#         validation_split_ratio = 0.10  # 90% for training, 10% for validation

#         datagenerator_kwargs = dict(
#             rescale=1.0 / 255,
#             validation_split=validation_split_ratio
#         )

#         dataflow_kwargs = dict(
#             target_size=tuple(self.config.params_image_size[:-1]),
#             batch_size=self.config.params_batch_size,
#             interpolation="bilinear",
#             class_mode='categorical'
#         )

#         # Create a single instance of ImageDataGenerator for both training and validation with optional augmentation
#         datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
#         if self.config.params_is_augmentation:
#             datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#                 rotation_range=40,
#                 horizontal_flip=True,
#                 width_shift_range=0.2,
#                 height_shift_range=0.2,
#                 shear_range=0.2,
#                 zoom_range=0.2,
#                 **datagenerator_kwargs
#             )

#         self.train_generator = datagenerator.flow_from_directory(
#             directory=self.config.training_data,
#             subset="training",
#             shuffle=True,
#             **dataflow_kwargs
#         )
#         self.valid_generator = datagenerator.flow_from_directory(
#             directory=self.config.training_data,
#             subset="validation",
#             shuffle=False,
#             **dataflow_kwargs
#         )

#         # Log the sample counts to verify the split
#         print(f"Training samples: {self.train_generator.samples}")
#         print(f"Validation samples: {self.valid_generator.samples}")

#     @staticmethod
#     def save_model(path: Path, model: tf.keras.Model):
#         """
#         Save the trained model to the specified path.
#         """
#         try:
#             model.save(path)
#             print(f"Model saved at {path}")
#         except Exception as e:
#             raise ValueError(f"Error saving model at {path}: {e}")

#     def train(self):
#         """
#         Train the model using the data generators.
#         """
#         if not self.model:
#             raise ValueError("Base model is not loaded. Call get_base_model first.")

#         if not self.train_generator or not self.valid_generator:
#             raise ValueError("Data generators are not initialized. Call train_valid_generator first.")

#         self.model.compile(
#             optimizer=tf.keras.optimizers.Adam(),
#             loss='categorical_crossentropy',
#             metrics=['accuracy']
#         )

#         self.model.fit(
#             self.train_generator,
#             epochs=self.config.params_epochs,
#             steps_per_epoch=self.train_generator.samples // self.train_generator.batch_size,
#             validation_data=self.valid_generator,
#             validation_steps=self.valid_generator.samples // self.valid_generator.batch_size,
#             verbose=1
#         )

#         self.save_model(self.config.trained_model_path, self.model)



import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        """
        Initializes the Training class with the provided configuration settings.
        """
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None

    def get_base_model(self):
        """
        Load the base model from the configured path.
        """
        try:
            self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
            print(f"Base model loaded from {self.config.updated_base_model_path}")
        except Exception as e:
            raise ValueError(f"Error loading model from {self.config.updated_base_model_path}: {e}")

    def train_valid_generator(self):
        """
        Set up the training and validation data generators, ensuring the training dataset
        is significantly larger than the validation dataset.
        """
        if not self.config.training_data.exists():
            raise FileNotFoundError(f"Training data directory does not exist: {self.config.training_data}")

        validation_split_ratio = 0.10  # 90% for training, 10% for validation

        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=validation_split_ratio
        )

        dataflow_kwargs = dict(
            target_size=tuple(self.config.params_image_size[:-1]),
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode='categorical'
        )

        # Create a single instance of ImageDataGenerator for both training and validation
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        # Use the same generator configuration for validation
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Log the sample counts and class indices to verify the split and labeling
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.valid_generator.samples}")
        print("Training class indices:", self.train_generator.class_indices)
        print("Validation class indices:", self.valid_generator.class_indices)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the trained model to the specified path.
        """
        try:
            model.save(path)
            print(f"Model saved at {path}")
        except Exception as e:
            raise ValueError(f"Error saving model at {path}: {e}")

    def train(self):
        """
        Train the model using the data generators.
        """
        if not self.model:
            raise ValueError("Base model is not loaded. Call get_base_model first.")

        if not self.train_generator or not self.valid_generator:
            raise ValueError("Data generators are not initialized. Call train_valid_generator first.")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.train_generator.samples // self.train_generator.batch_size,
            validation_data=self.valid_generator,
            validation_steps=self.valid_generator.samples // self.valid_generator.batch_size,
            verbose=1
        )

        self.save_model(self.config.trained_model_path, self.model)