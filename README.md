# Heart Disease Prediction using Logistic Regression
📘 Overview

This project predicts the likelihood of heart disease in a patient based on medical attributes such as age, cholesterol, blood pressure, and exercise history.
It uses Logistic Regression, a popular classification algorithm in machine learning.

🧠 Objective

To build a machine learning model that can classify whether a person is at risk (1) or not at risk (0) of heart disease.

📊 Dataset

File: heart.csv

Column Name	Description
Age	Age of the patient
Sex	Gender (1 = Male, 0 = Female)
ChestPainType	Type of chest pain (ATA, NAP, ASY, etc.)
RestingBP	Resting blood pressure (mm Hg)
Cholesterol	Serum cholesterol (mg/dl)
FastingBS	Fasting blood sugar (1 = >120 mg/dl, 0 = otherwise)
RestingECG	Resting electrocardiogram results
MaxHR	Maximum heart rate achieved
ExerciseAngina	Exercise-induced angina (Y/N)
Oldpeak	ST depression induced by exercise
ST_Slope	Slope of the ST segment
HeartDisease	Target variable (1 = disease, 0 = healthy)
🧩 Model Used

Logistic Regression

A statistical model used for binary classification.

Works well for medical and biological data.

⚙️ Steps in the Project

Importing libraries

Loading the dataset (heart.csv)

Data preprocessing

Splitting data into training and test sets

Training the Logistic Regression model

Evaluating the model performance

Making predictions on new patient data

🧾 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

🧮 Example Prediction
new_data = pd.DataFrame({
    "Age": [45],
    "Sex": [1],
    "ChestPainType": ["NAP"],
    "RestingBP": [130],
    "Cholesterol": [230],
    "FastingBS": [0],
    "RestingECG": ["Normal"],
    "MaxHR": [150],
    "ExerciseAngina": ["N"],
    "Oldpeak": [1.0],
    "ST_Slope": ["Up"]
})
prediction = model.predict(new_data)
print("Predicted Outcome (1=Heart Disease, 0=Healthy):", prediction[0])

💾 Requirements

Install the dependencies:

pip install pandas scikit-learn

📈 Results

The model achieved an accuracy of around 80–85%, depending on dataset quality and preprocessing.

🌟 Author

birdygirl-journey
🚀 Machine Learning Projects Collection
