import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF

# Sample dataset (Simplified NER-like format)
X = [[{'word': 'John'}, {'word': 'loves'}, {'word': 'Python'}],
     [{'word': 'Alice'}, {'word': 'codes'}, {'word': 'in'}, {'word': 'Java'}]]

y = [['B-PER', 'O', 'B-LANG'],  # Named Entity Recognition (NER) Labels
     ['B-PER', 'O', 'O', 'B-LANG']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Define CRF model
crf = CRF(algorithm='lbfgs', max_iterations=50)
crf.fit(X_train, y_train)

# Make predictions
y_pred = crf.predict(X_test)

# Flatten lists for confusion matrix
y_test_flat = [label for seq in y_test for label in seq]
y_pred_flat = [label for seq in y_pred for label in seq]

# Get unique labels
labels = list(set(y_test_flat + y_pred_flat))

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test_flat, y_pred_flat, labels=labels)

# Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)
plt.title("Confusion Matrix for CRF Predictions")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()