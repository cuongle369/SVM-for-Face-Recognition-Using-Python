import os
import cv2
from skimage.feature import hog
import pickle
import argparse

def extract_hog_features(image):
    features = hog(image, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=False)
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="face_data", help="Base directory containing person subdirectories")
    parser.add_argument("--output_file", default="features.pickle", help="Output file to save features and labels")
    args = parser.parse_args()

    features = []
    labels = []
    for person in os.listdir(args.base_dir):
        person_dir = os.path.join(args.base_dir, person)
        if os.path.isdir(person_dir):
            for image_file in os.listdir(person_dir):
                if image_file.endswith(".jpg"):
                    image_path = os.path.join(person_dir, image_file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        feat = extract_hog_features(image)
                        features.append(feat)
                        labels.append(person)

    with open(args.output_file, 'wb') as f:
        pickle.dump((features, labels), f)

if __name__ == "__main__":
    main()