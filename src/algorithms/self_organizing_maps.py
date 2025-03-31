import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from minisom import MiniSom  
from tensorflow.keras.datasets import mnist

# Choose dataset: 'mnist', 'wine', 'customers'
DATASET = "mnist"  # Change to 'wine' or 'customers' for different datasets

# Load dataset
if DATASET == "mnist":
    (X_train, y_train), _ = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # Flatten and normalize
    X_train, y_train = X_train[:1000], y_train[:1000]  # Use subset for faster training

elif DATASET == "wine":
    df = sns.load_dataset("wine_quality")
    X_train = df.drop(columns=["quality"]).values
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)  # Normalize
    y_train = df["quality"].values

elif DATASET == "customers":
    url = "https://raw.githubusercontent.com/MachineLearningWithPython/datasets/main/Mall_Customers.csv"
    df = pd.read_csv(url)
    X_train = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    y_train = None  # No predefined labels

# Initialize and train SOM
som_size = (10, 10)
som = MiniSom(som_size[0], som_size[1], X_train.shape[1], sigma=1.0, learning_rate=0.5)
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
    label = str(y_train[i]) if y_train is not None else "•"
    plt.text(winner[0], winner[1], label, color="black", fontsize=8, ha="center", va="center",
             bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))

plt.title(f"SOM Clustering - {DATASET.upper()} Dataset")
plt.xticks(range(som_size[0]))
plt.yticks(range(som_size[1]))
plt.grid(color="black", linestyle="--", linewidth=0.5)
plt.show()