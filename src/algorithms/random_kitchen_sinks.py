import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler

# Generate a non-linearly separable dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# Apply Random Kitchen Sinks (RKS) for kernel approximation
rks = RBFSampler(gamma=1.0, n_components=500, random_state=42)
X_rks = rks.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_rks)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', alpha=0.6)
plt.title("Data Transformed by Random Kitchen Sinks (RKS)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Make sure X is a numpy array before accessing shape
X_shape = X.shape if hasattr(X, 'shape') else np.array(X).shape
X_rks_shape = X_rks.shape if hasattr(X_rks, 'shape') else np.array(X_rks).shape

print("Original Data Shape:", X_shape)
print("Transformed Data Shape (RKS Features):", X_rks_shape)