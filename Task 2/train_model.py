import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the cleaned dataset
data = pd.read_csv("cleaned_ev_dataset.csv")

# Display basic dataset information
print("Dataset Shape:", data.shape)
print(data.head(5))
print(data.columns.tolist())

# Check target class distribution
print("\nGrid Status Distribution:")
print(data["Grid_Status_Label"].value_counts())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Feature Selection
features = [
    "Charger_Type",
    "Energy_kWh",
    "Duration_Mins",
    "Battery_Temp_C",
    "Peak_Hour_Flag",
    "Power_kW"
]

# Input features
X = data[features]

# Target variable
y = data["Grid_Status_Label"]

print("\nSelected Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# Step 2: One-Hot Encoding

# Define categorical and numerical features
categorical_features = ["Charger_Type"]

numerical_features = [
    "Energy_kWh",
    "Duration_Mins",
    "Battery_Temp_C",
    "Peak_Hour_Flag",
    "Power_kW"
]

# Create preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features),
        ("numerical", "passthrough", numerical_features)
    ]
)

print("\nPreprocessor configured successfully.")


# Step 3: Train/Test Split (80% / 20%)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% testing, 80% training
    random_state=42,     # Ensure reproducible results
    stratify=y           # Maintain class distribution
)


print("\nTraining Set:", X_train.shape)  # 80% of features
print("Testing Set:", X_test.shape)     # 20% of features

print("\nTraining Labels:", y_train.shape)  # Training targets
print("Testing Labels:", y_test.shape)     # Testing targets


# Step 4: Model Training

# Random Forest model
rf_model = Pipeline([
    ("preprocessor", clone(preprocessor)),  # Apply One-Hot Encoding
    ("classifier", RandomForestClassifier(
        n_estimators=100,                  # Use 100 decision trees
        random_state=42                    # Ensure reproducible results
    ))
])

# Decision Tree model
dt_model = Pipeline([
    ("preprocessor", clone(preprocessor)),  # Apply One-Hot Encoding
    ("classifier", DecisionTreeClassifier(
        random_state=42                    # Ensure reproducible results
    ))
])

# Train Random Forest
rf_model.fit(X_train, y_train)  # Learn patterns from training data
print("Random Forest training completed.")

# Train Decision Tree
dt_model.fit(X_train, y_train)  # Learn patterns from training data
print("Decision Tree training completed.")


# Step 5: Model Evaluation & Comparison

# Store the trained models
models = {
    "Random Forest": rf_model,
    "Decision Tree": dt_model
}

results = []  # Store evaluation results

for name, model in models.items():

    # Predict using the testing dataset
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)  # Accuracy
    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0 # Weighted Precision
    )
    recall = recall_score(
        y_test, y_pred, average="weighted", zero_division=0 # Weighted Precision
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0 # Weighted F1-score
    )

    # Store results for model comparison
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    # Show detailed performance for each class
    print(f"\n{name} Classification Report:")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

# Create a comparison table
results_df = pd.DataFrame(results)

# Rank models by F1-score (highest first)
results_df = results_df.sort_values(
    by="F1-score", ascending=False
).reset_index(drop=True)

print("\nModel Comparison:")
print(results_df.round(4).to_string(index=False))

# Select the best model based on weighted F1-score
best_model_name = results_df.loc[0, "Model"]
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)


# Step 6: Confusion Matrix Visualization

# Define the target class labels
labels = ["Normal", "Overheated", "Grid_Surge"]

# Generate a confusion matrix for each model
for name, model in models.items():

    y_pred = model.predict(X_test)  # Predict test data

    cm = confusion_matrix(
        y_test, y_pred, labels=labels
    )  # Compare actual and predicted labels

    plt.figure(figsize=(7, 5))  # Set figure size

    sns.heatmap(
        cm,
        annot=True,               # Display numbers in cells
        fmt="d",                  # Show integer values
        cmap="Blues",             # Use blue color
        xticklabels=labels,       # Predicted class labels
        yticklabels=labels        # Actual class labels
    )

    plt.title(f"{name} - Confusion Matrix")  # Title
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    # Save the confusion matrix as a PNG file
    filename = name.lower().replace(" ", "_")
    plt.savefig(f"{filename}_confusion_matrix.png", dpi=300)

    plt.show()
    plt.close()

    # Step 7: Save the Best Model

    # Save the best model
    joblib.dump(best_model, "ev_model.pkl")  # Export model

    print("\nBest Model:", best_model_name)
    print("Model saved successfully as ev_model.pkl")



