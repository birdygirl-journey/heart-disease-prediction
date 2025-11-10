# Diabetes Prediction using Logistic Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("diabetes.csv")
print(df.head())

# Split into features and target
X = df.drop(columns="Outcome")
y = df["Outcome"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Example prediction
new_data = pd.DataFrame({
    "Pregnancies": [2],
    "Glucose": [120],
    "BloodPressure": [70],
    "SkinThickness": [25],
    "Insulin": [80],
    "BMI": [30.5],
    "DiabetesPedigreeFunction": [0.5],
    "Age": [29]
})
prediction = model.predict(new_data)
print("Predicted Outcome (1=Diabetic, 0=Non-Diabetic):", prediction[0])
