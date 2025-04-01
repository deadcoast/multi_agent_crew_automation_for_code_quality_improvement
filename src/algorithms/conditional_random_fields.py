import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Add try-except for potentially missing sklearn_crfsuite with dynamic import
sklearn_crfsuite_module = None
try:
    import importlib
    sklearn_crfsuite_module = importlib.import_module('sklearn_crfsuite')
    print("Successfully imported sklearn_crfsuite")
except ImportError:
    print("Warning: sklearn_crfsuite not installed. Please install with: pip install sklearn-crfsuite")

# Sample dataset (Simplified NER-like format)
X = [
    [{"word": "John"}, {"word": "loves"}, {"word": "Python"}],
    [{"word": "Alice"}, {"word": "codes"}, {"word": "in"}, {"word": "Java"}],
]

y = [
    ["B-PER", "O", "B-LANG"],  # Named Entity Recognition (NER) Labels
    ["B-PER", "O", "O", "B-LANG"],
]

# Only run the rest of the code if the imports succeeded
def run_crf_example():
    if sklearn_crfsuite_module is None:
        print("Cannot run example: sklearn_crfsuite is not installed")
        return
        
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # Define CRF model
    crf = sklearn_crfsuite_module.CRF(algorithm="lbfgs", max_iterations=50)
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

# Only run if this module is called directly
if __name__ == "__main__":
    try:
        run_crf_example()
    except Exception as e:
        print(f"Error running CRF example: {str(e)}")
        if "sklearn_crfsuite" in str(e):
            print("Please install sklearn_crfsuite: pip install sklearn-crfsuite")
