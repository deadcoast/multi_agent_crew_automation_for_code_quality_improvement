from typing import Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from minisom import MiniSom
from sklearn import datasets
from sklearn.utils import Bunch

# Choose dataset: 'mnist', 'wine', 'customers'
DATASET = "mnist"  # Change to 'wine' or 'customers' for different datasets

# Initialize variables to avoid "possibly unbound" errors
X_train = np.array([]).reshape(0, 1)  # Empty array with shape (0, 1) for initial value
y_train = np.array([])  # Empty array, not None

# Load dataset
if DATASET == "mnist":
    # Load sklearn's digits dataset instead of TensorFlow's MNIST
    digits: Bunch = cast(Bunch, datasets.load_digits())
    X_train = digits.images.reshape(digits.images.shape[0], -1) / 16.0  # Flatten and normalize
    y_train = digits.target
    X_train, y_train = X_train[:1000], y_train[:1000]  # Use subset for faster training

elif DATASET == "wine":
    df = sns.load_dataset("wine_quality")
    X_train = df.drop(columns=["quality"]).values
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)  # Normalize
    y_train = df["quality"].values

elif DATASET == "customers":
    url = "https://raw.githubusercontent.com/MachineLearningWithPython/datasets/main/Mall_Customers.csv"
    df = pd.read_csv(url)
    X_train = np.array(df[["Annual Income (k$)", "Spending Score (1-100)"]].values)
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    # For customers dataset, create a dummy label array of zeros
    y_train = np.array([0] * len(X_train))

# Initialize and train SOM
som_size: Tuple[int, int] = (10, 10)
# Convert float to int for sigma parameter
som = MiniSom(som_size[0], som_size[1], X_train.shape[1], sigma=1, learning_rate=0.5)
som.random_weights_init(X_train)
som.train_random(X_train, 1000)

# Create activity heatmap
activation_map = np.zeros(som_size)
for x in X_train:
    winner = som.winner(x)
    activation_map[winner] += 1

plt.figure(figsize=(10, 8))
plt.imshow(activation_map.T, cmap="coolwarm", origin="lower", alpha=0.7)
plt.colorbar(label="Neuron Activation Frequency")

# Overlay Data Points
for i, x in enumerate(X_train):
    winner = som.winner(x)
    # Check if we should use a label from y_train or a default marker
    label = "•" if DATASET == "customers" else str(y_train[i])
    # Convert numpy.intp to float for plt.text coordinates
    plt.text(
        float(winner[0]),
        float(winner[1]),
        label,
        color="black",
        fontsize=8,
        ha="center",
        va="center",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )

plt.title(f"SOM Clustering - {DATASET.upper()} Dataset")
plt.xticks(range(som_size[0]))
plt.yticks(range(som_size[1]))
plt.grid(color="black", linestyle="--", linewidth=0.5)
plt.show()
