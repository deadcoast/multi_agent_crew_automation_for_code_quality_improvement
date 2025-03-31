import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

# Generate Synthetic Data (Normal Data)
rng = np.random.RandomState(42)
X = 0.3 * rng.randn(100, 2)  # 100 normal points

# Add Some Anomalies (Outliers)
X_outliers = rng.uniform(low=-4, high=4, size=(10, 2))  # 10 outliers

# Combine Normal Data & Outliers
X = np.vstack([X, X_outliers])


iso_forest = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
y_pred = iso_forest.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Isolation Forest Anomaly Detection")
plt.show()