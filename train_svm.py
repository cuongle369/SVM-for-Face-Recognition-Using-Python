import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import preprocessing
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="features.pickle", help="Input file containing features and labels")
    parser.add_argument("--output_model", default="svm_model.joblib", help="Output file for the trained SVM model")
    parser.add_argument("--output_label_encoder", default="label_encoder.joblib", help="Output file for the label encoder")
    args = parser.parse_args()

    with open(args.input_file, 'rb') as f:
        features, labels = pickle.load(f)

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(labels)
    y_encoded = label_encoder.transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, y_encoded, test_size=0.2, random_state=42)

    svm = SVC(kernel='rbf', gamma='auto')
    svm.fit(X_train, y_train)

    accuracy = svm.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    joblib.dump(svm, args.output_model)
    joblib.dump(label_encoder, args.output_label_encoder)

if __name__ == "__main__":
    main()