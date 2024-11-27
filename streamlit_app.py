import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Water Potability Prediction App")

# Sidebar
st.sidebar.header("Options")

# Load Dataset
@st.cache
def load_data():
    url = "https://www.kaggleusercontent.com/datasets/adityakadiwal/water-potability/water_potability.csv"
    df = pd.read_csv(url)
    return df

data = load_data()

# Data Preprocessing
st.header("Data Preprocessing")
data['Potability'] = data['Potability'].astype('category')
for col in data.columns:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].median(), inplace=True)
st.write("Dataset after filling missing values:")
st.dataframe(data.head())

# Data Splitting
st.header("Train-Test Split")
st.write("Splitting the dataset into training and testing sets (70%-30%)")
X = data.drop('Potability', axis=1)
y = data['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
st.write(f"Training set: {X_train.shape[0]} samples")
st.write(f"Testing set: {X_test.shape[0]} samples")

# Model Training
st.header("Model Training")
clf = DecisionTreeClassifier(random_state=42, min_samples_split=10, min_samples_leaf=20)
clf.fit(X_train, y_train)
st.write("Decision Tree trained!")

# Model Visualization
st.subheader("Decision Tree Structure")
tree_rules = export_text(clf, feature_names=list(X.columns))
st.text(tree_rules)

# Model Evaluation
st.header("Model Evaluation")
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)

st.write("Confusion Matrix:")
cm = confusion_matrix(y_test, test_pred)
st.write(cm)

st.write("Classification Report:")
report = classification_report(y_test, test_pred, target_names=['Not Potable', 'Potable'])
st.text(report)

accuracy = accuracy_score(y_test, test_pred)
st.write(f"Accuracy on test data: {accuracy:.2f}")

# Feature Importance
st.header("Feature Importance")
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
st.bar_chart(importances)

# Predictions on New Samples
st.header("Try Predictions on New Water Samples")
def get_sample_input():
    sample = {}
    for col in X.columns:
        sample[col] = st.number_input(f"{col}", value=float(data[col].median()))
    return pd.DataFrame([sample])

sample_input = get_sample_input()
st.write("Sample Input:")
st.dataframe(sample_input)
if st.button("Predict Potability"):
    prediction = clf.predict(sample_input)[0]
    result = "Potable" if prediction == 1 else "Not Potable"
    st.write(f"Prediction: {result}")
