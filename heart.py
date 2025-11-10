# Heart Disease Prediction using Logistic Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("heart.csv")
print("Dataset preview:")
print(df.head())

# Split data into features and target
X = df.drop(columns="target")
y = df["target"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example prediction
new_data = pd.DataFrame({
    "age": [54],
    "sex": [1],
    "cp": [0],
    "trestbps": [140],
    "chol": [239],
    "fbs": [0],
    "restecg": [1],
    "thalach": [160],
    "exang": [0],
    "oldpeak": [1.2],
    "slope": [2],
    "ca": [0],
    "thal": [2]
})

prediction = model.predict(new_data)
print("\nPredicted Outcome (1 = Heart Disease, 0 = No Heart Disease):", prediction[0])
