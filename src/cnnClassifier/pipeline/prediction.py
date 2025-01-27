import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model = load_model("/Users/tejasredkar/Developer/Chest-Cancer-Prediction-using-ML-Flow-DVC/artifacts/training/model.h5")

        # Load and preprocess image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image) / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        # Make prediction
        probs = model.predict(test_image)
        print("Class probabilities:", probs)

        # Match the order from training: 0 => Adenocarcinoma, 1 => Normal
        class_labels = ["Adenocarcinoma Cancer", "Normal"]

        # Print out probabilities for each class
        for i, prob in enumerate(probs[0]):
            print(f"{class_labels[i]}: {prob:.4f}")

        # If the probability of Adenocarcinoma is >= threshold, interpret it accordingly
        threshold = 0.5
        prediction = "Adenocarcinoma Cancer" if probs[0][0] >= threshold else "Normal"
        print(f"Final prediction: {prediction}")

        return [{"image": prediction}]