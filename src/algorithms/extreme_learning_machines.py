from typing import cast

import hpelm  # High-Performance ELM
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

# Load dataset
iris: Bunch = cast(Bunch, datasets.load_iris())
X = np.array(iris.data)  # Convert to numpy array explicitly
y = np.array(iris.target)  # Convert to numpy array explicitly
target_names = iris.target_names  # Store target names separately

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Convert labels to one-hot encoding (for ELM)
y_train_onehot = np.eye(len(set(y)))[y_train]
y_test_onehot = np.eye(len(set(y)))[y_test]

# Define and Train ELM - ensure we use numpy arrays with shape attribute
X_train_np = np.array(X_train)
y_train_onehot_np = np.array(y_train_onehot)

elm = hpelm.ELM(X_train_np.shape[1], y_train_onehot_np.shape[1])
elm.add_neurons(50, "sigm")  # 50 hidden neurons with sigmoid activation
elm.train(X_train_np, y_train_onehot_np, "c")  # 'c' for classification

# Make predictions
y_pred = elm.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {accuracy:.2f}")

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Normalize

# Create the figure
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm_percentage,
    annot=True,
    fmt=".2%",
    cmap="coolwarm",
    linewidths=2,
    xticklabels=target_names,
    yticklabels=target_names,
)

# Improve readability
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("ELM Confusion Matrix (Normalized)", fontsize=14)
plt.show()
