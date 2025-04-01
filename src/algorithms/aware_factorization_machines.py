import math
import random

import numpy as np


class FFMModel:
    def __init__(self, n_features, n_fields, k_factors, eta, lambda_reg, n_epochs):
        """
        Initializes the FFM model.

        Args:
            n_features (int): Total number of features (n).
            n_fields (int): Total number of fields (f).
            k_factors (int): Number of latent factors (k).
            eta (float): Learning rate.
            lambda_reg (float): Regularization parameter.
            n_epochs (int): Number of training epochs.
        """
        self.n = n_features
        self.f = n_fields
        self.k = k_factors
        self.eta = eta
        self.lambda_reg = lambda_reg
        self.n_epochs = n_epochs

        # Initialize model parameters w: n x f x k tensor
        # Samples from U(0, 1/sqrt(k)) [cite: 82]
        self.w = np.random.uniform(0, 1.0 / math.sqrt(self.k), (self.n, self.f, self.k))

        # Initialize sum of squared gradients G: n x f x k tensor
        # Initialized to ones [cite: 80, 83]
        self.G = np.ones((self.n, self.f, self.k))

    def predict_phi(self, x_sparse):
        """
        Calculates the FFM output phi for a given sparse input instance.

        Args:
            x_sparse (list): A list of tuples (field_index, feature_index, value).
                             Example: [(0, 10, 1.0), (1, 156, 0.5), (1, 157, 0.5)]

        Returns:
            float: The predicted phi value.
        """
        phi = 0.0
        num_non_zero = len(x_sparse)
        for i in range(num_non_zero):
            for j in range(i + 1, num_non_zero):
                f1, j1, v1 = x_sparse[i]
                f2, j2, v2 = x_sparse[j]

                # Get relevant latent vectors w_j1,f2 and w_j2,f1
                # Ensure indices are within bounds
                if (
                    0 <= j1 < self.n
                    and 0 <= f2 < self.f
                    and 0 <= j2 < self.n
                    and 0 <= f1 < self.f
                ):
                    w_j1_f2 = self.w[j1, f2, :]
                    w_j2_f1 = self.w[j2, f1, :]

                    # Calculate dot product
                    dot_product = np.dot(w_j1_f2, w_j2_f1)
                    phi += dot_product * v1 * v2
        return phi

    def train(self, S_sparse):
        """
        Trains the FFM model using Stochastic Gradient Descent with AdaGrad.

        Args:
            S_sparse (list): Training data, list of tuples (y, x_sparse).
                           y is the label (e.g., -1 or 1 for logistic loss).
                           x_sparse is a list of (field_idx, feature_idx, value).
        """
        for epoch in range(self.n_epochs):
            print(f"Starting Epoch {epoch + 1}/{self.n_epochs}")
            random.shuffle(S_sparse)  # Process data in random order

            for y, x_sparse in S_sparse:
                # Ensure label y is typically -1 or 1 for this loss formulation
                if y == 0:
                    y = -1  # Convert 0/1 labels if needed

                # 1. Calculate phi [cite: 73]
                phi = self.predict_phi(x_sparse)

                # 2. Calculate kappa [cite: 79]
                # Add small epsilon to avoid potential exp overflow/underflow
                exp_term = math.exp(y * phi)
                kappa = -y / (1.0 + exp_term)

                # 3. Iterate through pairs of non-zero features [cite: 80]
                num_non_zero = len(x_sparse)
                for i in range(num_non_zero):
                    for j in range(i + 1, num_non_zero):
                        f1, j1, v1 = x_sparse[i]
                        f2, j2, v2 = x_sparse[j]

                        # Ensure indices are within bounds before accessing w and G
                        if not (
                            0 <= j1 < self.n
                            and 0 <= f2 < self.f
                            and 0 <= j2 < self.n
                            and 0 <= f1 < self.f
                        ):
                            # print(f"Warning: Skipping update due to out-of-bounds index - j1:{j1}, f2:{f2} or j2:{j2}, f1:{f1}")
                            continue

                        w_j1_f2 = self.w[j1, f2, :]
                        w_j2_f1 = self.w[j2, f1, :]

                        # 4. Calculate sub-gradients [cite: 79]
                        g_j1_f2 = self.lambda_reg * w_j1_f2 + kappa * w_j2_f1 * v1 * v2
                        g_j2_f1 = self.lambda_reg * w_j2_f1 + kappa * w_j1_f2 * v1 * v2

                        # 5. Update G and w for each dimension d [cite: 80, 81]
                        self.G[j1, f2, :] += g_j1_f2**2
                        self.G[j2, f1, :] += g_j2_f1**2

                        self.w[j1, f2, :] -= (
                            self.eta / np.sqrt(self.G[j1, f2, :])
                        ) * g_j1_f2
                        self.w[j2, f1, :] -= (
                            self.eta / np.sqrt(self.G[j2, f1, :])
                        ) * g_j2_f1

            # Optional: Calculate loss on validation set for early stopping here
            print(f"Epoch {epoch + 1} completed.")

        print("Training finished.")


# --- Example Usage ---
if __name__ == "__main__":
    # --- Parameters ---
    N_FEATURES = 1000  # Example: Total unique features after hashing/encoding
    N_FIELDS = 10  # Example: Number of distinct fields
    K_FACTORS = 4  # Example: Number of latent factors [cite: 202] (Value from paper)
    ETA = 0.1  # Example: Learning rate [cite: 160] (Value from paper experiments)
    LAMBDA_REG = 2e-5  # Example: Regularization [cite: 202] (Value from paper)
    N_EPOCHS = 10  # Example: Number of epochs [cite: 80] (Adjust as needed)

    # --- Generate Synthetic Sparse Data ---
    # Format: [(y, [(field_idx, feature_idx, value), ...]), ...]
    # Ensure feature_idx < N_FEATURES and field_idx < N_FIELDS
    DATA_SIZE = 1000
    MAX_NON_ZERO_PER_INSTANCE = 15
    train_data_sparse = []
    for _ in range(DATA_SIZE):
        y_label = random.choice([-1, 1])  # Use -1/1 for logistic loss
        num_non_zero = random.randint(5, MAX_NON_ZERO_PER_INSTANCE)
        x_instance_sparse = []
        used_features = set()
        for _ in range(num_non_zero):
            field = random.randint(0, N_FIELDS - 1)
            # Ensure unique feature index per instance for simplicity here
            feature = random.randint(0, N_FEATURES - 1)
            while feature in used_features:
                feature = random.randint(0, N_FEATURES - 1)
            used_features.add(feature)

            # Often values are 1 for categorical features after one-hot encoding
            value = 1.0
            x_instance_sparse.append((field, feature, value))
        train_data_sparse.append((y_label, x_instance_sparse))

    print(f"Generated {len(train_data_sparse)} training instances.")
    print(
        f"Example instance: y={train_data_sparse[0][0]}, x={train_data_sparse[0][1][:3]}..."
    )  # Show first 3 features

    # --- Initialize and Train ---
    ffm = FFMModel(N_FEATURES, N_FIELDS, K_FACTORS, ETA, LAMBDA_REG, N_EPOCHS)
    ffm.train(train_data_sparse)

    # --- Make a Prediction (Example) ---
    if train_data_sparse:
        example_y, example_x = train_data_sparse[0]
        predicted_phi = ffm.predict_phi(example_x)
        # Convert phi to probability if needed (e.g., sigmoid)
        predicted_prob = 1.0 / (1.0 + math.exp(-predicted_phi))
        print("\nExample Prediction:")
        print("  Instance features (first 3): {}...".format(example_x[:3]))
        print("  True Label (y): {}".format(example_y))
        print(f"  Predicted Phi: {predicted_phi:.4f}")
        print(f"  Predicted Probability (Sigmoid): {predicted_prob:.4f}")
