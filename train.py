import os
import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# Load extracted features and labels
def load_features_labels(feature_file):
    with open(feature_file, 'rb') as f:
        features, labels = pickle.load(f)
    return features, labels


# Train SVM classifier with GridSearch
def train_svm(X_train, y_train):
    param_grid = {'C': [0.1], 'gamma': [1], 'kernel': ['linear']}
    svm = SVC(probability=True)
    grid = GridSearchCV(svm, param_grid, refit=True, verbose=2, n_jobs=8, cv=5)
    grid.fit(X_train, y_train)

    return grid


# Evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    return accuracy, report, cm


# Save trained model and evaluation results
def save_results(model, accuracy, report, confusion_mtx, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Save best model
    model_file = os.path.join(output_dir, "best_svm_model.pickle")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    # Save classification report and best parameters
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f'Accuracy: {accuracy:.2f}\n')
        f.write(report)
        f.write("\nBest Parameters:\n")
        f.write(str(model.best_params_))

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

    print("Best Parameters:", model.best_params_)


# Main
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train SVM with GridSearch and save results.")
    parser.add_argument("--features_file", default="features.pickle", help="File containing features and labels")
    parser.add_argument("--output_dir", default="svm_results", help="Output directory to save trained SVM and results")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of data to use for testing (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    features, labels = load_features_labels(args.features_file)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=args.test_size, random_state=args.random_state, stratify=labels
    )

    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # Train model on training data only
    model = train_svm(X_train, y_train)

    # Evaluate model on testing data
    accuracy, report, cm = evaluate_model(model, X_test, y_test)

    print(f"Test accuracy: {accuracy:.4f}")

    # Save results
    save_results(model, accuracy, report, cm, args.output_dir)

    print(f"Model and evaluation results saved to {args.output_dir}")
    
    with open('best_svm_model.pickle', 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved")


if __name__ == "__main__":
    main()
