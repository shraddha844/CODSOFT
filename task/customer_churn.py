import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------
# 1. Load Dataset
# ---------------------------------------------
FILE_PATH = "churn.csv"     # Make sure churn.csv is in same folder

df = pd.read_csv(FILE_PATH)

print("Dataset Loaded Successfully!")
print(df.head())

# ---------------------------------------------
# 2. Encode Categorical Columns
# ---------------------------------------------
le = LabelEncoder()

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ---------------------------------------------
# 3. Features and Target
# ---------------------------------------------
# Works for Kaggle "Customer Churn" dataset where target column is "Churn"
# If your file uses "Exited" or something else, change below:
target_column = "Churn"

X = df.drop(target_column, axis=1)
y = df[target_column]

# ---------------------------------------------
# 4. Train-Test Split
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------
# 5. Random Forest Classifier
# ---------------------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------
# 6. Predictions & Accuracy
# ---------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------------------------
# 7. Predict a Single New Customer (Example)
# ---------------------------------------------
print("\n--- Predicting for New Example Customer ---")

example = X.iloc[0]     # predict using first row of dataset
prediction = model.predict([example])

print("Prediction for sample customer:", "Churn" if prediction[0] == 1 else "Not Churn")