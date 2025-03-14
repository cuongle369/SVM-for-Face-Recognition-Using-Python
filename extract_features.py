import os
import cv2
import numpy as np
from skimage.feature import hog
import pickle
import argparse
import matplotlib.pyplot as plt
from data_augmentation import extract_features_with_augmentation


def extract_hog_features(image, visualize=True):
    """
    Extract HOG features from an image and optionally return the visualization image.

    Parameters:
    - image: input image
    - visualize: if True, returns both the feature vector and the visualization image

    Returns:
    - features: HOG feature vector
    - hog_image: HOG visualization image (if visualize=True)
    """
    if visualize:
        features, hog_image = hog(image, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True)
        return features, hog_image
    else:
        features = hog(image, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)
        return features


def save_visualization(original_image, hog_image, output_path):
    """
    Save the original image and its HOG visualization side by side.

    Parameters:
    - original_image: the original image
    - hog_image: the HOG visualization image
    - output_path: path to save the output image
    """
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Display the original image
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Display the HOG visualization image
    ax2.imshow(hog_image, cmap='gray')
    ax2.set_title('HOG Features')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="face_data", help="Base directory containing person subdirectories")
    parser.add_argument("--output_file", default="features.pickle", help="Output file to save features and labels")
    parser.add_argument("--vis_dir", default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--use_augmentation", action="store_true", help="Apply data augmentation")
    args = parser.parse_args()

    # Create visualization directory if it does not exist
    os.makedirs(args.vis_dir, exist_ok=True)

    features_and_labels = []

    for person in os.listdir(args.base_dir):
        person_dir = os.path.join(args.base_dir, person)
        if os.path.isdir(person_dir):
            # Create a visualization directory for each person
            person_vis_dir = os.path.join(args.vis_dir, person)
            os.makedirs(person_vis_dir, exist_ok=True)

            for image_file in os.listdir(person_dir):
                if image_file.endswith(".jpg"):
                    image_path = os.path.join(person_dir, image_file)

                    if args.use_augmentation:
                        # Extract features with augmentation
                        augmented_features_labels = extract_features_with_augmentation(
                            image_path, person, person_vis_dir, visualize=True)
                        features_and_labels.extend(augmented_features_labels)
                        print(
                            f"Processed with augmentation: {image_file} (generated {len(augmented_features_labels)} samples)")
                    else:
                        # Original method without augmentation
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            # Extract HOG features with visualization enabled
                            feat, hog_image = extract_hog_features(image, visualize=True)

                            # Store features and labels
                            features_and_labels.append((feat, person))

                            # Save the visualization image
                            vis_filename = os.path.splitext(image_file)[0] + "_hog.png"
                            vis_path = os.path.join(person_vis_dir, vis_filename)
                            save_visualization(image, hog_image, vis_path)

                            print(f"Processed and visualized: {image_file}")

    # Unpack features and labels from the list of tuples
    features = [item[0] for item in features_and_labels]
    labels = [item[1] for item in features_and_labels]

    # Save features and labels to a pickle file
    with open(args.output_file, 'wb') as f:
        pickle.dump((features, labels), f)

    print(f"Features extracted and saved to {args.output_file}")
    print(f"Total samples after processing: {len(features)}")
    print(f"Visualizations saved to {args.vis_dir}")


if __name__ == "__main__":
    main()
