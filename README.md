# 🩺 End-to-End Chest Cancer Classification using MLflow and DVC

![GitHub last commit](https://img.shields.io/github/last-commit/tejred213/Chest-Cancer-Prediction-using-ML-Flow-DVC?style=for-the-badge&logo=github&logoColor=white)
![GitHub repo size](https://img.shields.io/github/repo-size/tejred213/Chest-Cancer-Prediction-using-ML-Flow-DVC?style=for-the-badge&logo=github&logoColor=white)
![GitHub issues](https://img.shields.io/github/issues/tejred213/Chest-Cancer-Prediction-using-ML-Flow-DVC?style=for-the-badge&logo=github&logoColor=white)
![GitHub license](https://img.shields.io/github/license/tejred213/Chest-Cancer-Prediction-using-ML-Flow-DVC?style=for-the-badge&logo=github&logoColor=white)

---

## 🌟 **Overview**

This project provides a complete solution for classifying chest cancer images using a **Convolutional Neural Network (CNN)**. Key features include:

- **MLflow**: Experiment tracking and model management.
- **DVC**: Data and model version control.
- **Docker & Flask**: Seamless deployment capabilities.

🔗 **Live Demo**: _Coming Soon_

---

## 📑 **Table of Contents**

1. [Project Highlights](#-project-highlights)
2. [Technologies Used](#-technologies-used)
3. [Setup Instructions](#-setup-instructions)
4. [Usage Guide](#-usage-guide)
5. [Model Training](#-model-training)
6. [Evaluation Metrics](#-evaluation-metrics)
7. [Deployment](#-deployment)
8. [Contributing](#-contributing)
9. [License](#-license)

---

## 🚀 **Project Highlights**

- **End-to-End Pipeline**: Includes everything from preprocessing to deployment.
- **Efficient Experimentation**: Leverages MLflow to track metrics and artifacts.
- **Version Control**: DVC ensures robust data and model tracking.
- **Extensible Design**: Modular architecture for adding features.
- **Deployment-Ready**: Pre-built Dockerized Flask app for production.

---

## 💻 **Technologies Used**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dvc&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

---

## 🛠️ **Setup Instructions**

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/DevanshuSurana/End-to-End-Chest-Cancer-Classification-using-MLflow-DVC.git
   cd Chest-Cancer-Prediction-using-ML-Flow-DVC
   ```

2. **Set up Virtual Environment (MacOS):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   For **Windows**:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Pull Data and Models:**

   ```bash
   dvc pull
   ```

5. **Configure MLflow:**

   Update `config.yaml` with your MLflow server details.

---

## 🧑‍🔬 **Usage Guide**

- **Prepare Data:**  
  Ensure your dataset follows the required structure and update paths in `params.yaml`.

- **Run Training Pipeline:**

  ```bash
  python main.py
  ```

- **Visualize Results:**  
  Track metrics and artifacts in the **MLflow UI**.

---

## 📊 **Model Training**

Execute the training pipeline:

```bash
python main.py
```

🎯 **Key Features:**

- Automated preprocessing.
- CNN model training.
- Experiment tracking with MLflow.

---

## 📈 **Evaluation Metrics**

Model performance metrics, such as **accuracy**, **precision**, and **recall**, are logged to:

- `scores.json`: A summary of evaluation results.
- MLflow UI: Visualize performance trends.

---

## 🚢 **Deployment**

Deploy the trained model using Flask:

```bash
python app.py
```

Access the web application at `http://localhost:5000`.

---

## 🤝 **Contributing**

We ❤️ contributions! Here's how you can help:

1. Fork the repository.
2. Create a new feature branch.
3. Submit a pull request.

---

## 📜 **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---
