import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model = load_model("/Users/tejasredkar/Developer/Chest-Cancer-Prediction-using-ML-Flow-DVC/artifacts/training/model.h5")

        # Load and preprocess image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image) / 255.0  # Normalize
        test_image = np.expand_dims(test_image, axis=0)

        # Make prediction
        probs = model.predict(test_image)
        print("Class probabilities:", probs)

        # Define class labels
        class_labels = ['Normal', 'Adenocarcinoma Cancer']
        for i, prob in enumerate(probs[0]):
            print(f"{class_labels[i]}: {prob:.4f}")

        # Adjusted threshold logic (if required)
        threshold = 0.5  # Default
        prediction = "Normal" if probs[0][0] >= threshold else "Adenocarcinoma Cancer"
        print(f"Final prediction: {prediction}")

        return [{"image": prediction}]

