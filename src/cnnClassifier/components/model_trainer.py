from dataclasses import dataclass
from pathlib import Path
import os
import tensorflow as tf
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (TrainingConfig)

class Training:
    def __init__(self, config: TrainingConfig):
        """
        Initializes the Training class with configuration settings.
        """
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None

    def load_base_model(self):
        """
        Load the base model from the specified path in the configuration.
        """
        try:
            self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
            print(f"Base model loaded from {self.config.updated_base_model_path}")
        except Exception as e:
            raise ValueError(f"Error loading model from {self.config.updated_base_model_path}: {e}")

    def train_valid_generator(self):
        """
        Set up the training and validation data generators.
        """
        if not self.config.training_data.exists():
            raise FileNotFoundError(f"Training data directory does not exist: {self.config.training_data}")

        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Validation generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=str(self.config.training_data),
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Training generator with augmentation if enabled
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
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=str(self.config.training_data),
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

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
        Train the model using the generators and save the trained model.
        """
        if not self.model:
            raise ValueError("Base model is not loaded. Call `load_base_model` first.")

        if not self.train_generator or not self.valid_generator:
            raise ValueError("Data generators are not initialized. Call `train_valid_generator` first.")

        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        print(f"Starting training for {self.config.params_epochs} epochs...")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)