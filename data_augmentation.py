import os
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import transform
from skimage.util import random_noise


def augment_image(image):
    """
    Generate augmented versions of an input image.

    Parameters:
    - image: grayscale input image

    Returns:
    - List of augmented images
    """
    augmented_images = []

    # Original image
    augmented_images.append(image)

    # Horizontal flip
    flipped = np.fliplr(image)
    augmented_images.append(flipped)

    # Rotation (+/- 10 degrees)
    for angle in [-10, 10]:
        rotated = transform.rotate(image, angle, resize=False, preserve_range=True).astype(np.uint8)
        augmented_images.append(rotated)

    # Brightness adjustments
    for factor in [0.8, 1.2]:
        brightness_adjusted = np.clip(image * factor, 0, 255).astype(np.uint8)
        augmented_images.append(brightness_adjusted)

    # Add noise
    noisy = random_noise(image, mode='gaussian', var=0.01)
    noisy = (noisy * 255).astype(np.uint8)
    augmented_images.append(noisy)

    # Slight zoom (crop center and resize)
    h, w = image.shape
    crop_percent = 0.9
    crop_h, crop_w = int(h * crop_percent), int(w * crop_percent)
    start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
    cropped = image[start_h:start_h + crop_h, start_w:start_w + crop_w]
    zoomed = cv2.resize(cropped, (h, w))
    augmented_images.append(zoomed)

    return augmented_images


def save_augmentation_examples(image_path, output_dir):
    """
    Save examples of data augmentation for visualization.

    Parameters:
    - image_path: path to the original image
    - output_dir: directory to save visualizations
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return

    augmented_images = augment_image(image)
    titles = ['Original', 'Flipped', 'Rotated (-10°)', 'Rotated (10°)',
              'Darker', 'Brighter', 'Noisy', 'Zoomed']

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()

    for i, (img, title) in enumerate(zip(augmented_images, titles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save visualization
    output_path = os.path.join(output_dir, "augmentation_examples.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Augmentation examples saved to {output_path}")


def extract_features_with_augmentation(image_path, person_name, output_dir=None, visualize=False):
    """
    Extract HOG features from original and augmented images.

    Parameters:
    - image_path: path to the original image
    - person_name: name of the person for labeling
    - output_dir: directory to save visualizations (if visualize=True)
    - visualize: whether to save visualization of HOG features

    Returns:
    - List of (features, label) tuples
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return []

    # Generate augmented images
    augmented_images = augment_image(image)

    # Extract features from all images
    result = []

    for i, aug_image in enumerate(augmented_images):
        # Extract HOG features
        if visualize and output_dir:
            features, hog_image = hog(aug_image, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2), visualize=True)

            # Save visualization if requested
            if i == 0:  # Only save visualization for original image
                basename = os.path.basename(image_path)
                vis_filename = os.path.splitext(basename)[0] + "_hog.png"
                vis_path = os.path.join(output_dir, vis_filename)

                # Create a figure with 2 subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.imshow(aug_image, cmap='gray')
                ax1.set_title('Original Image')
                ax1.axis('off')
                ax2.imshow(hog_image, cmap='gray')
                ax2.set_title('HOG Features')
                ax2.axis('off')
                plt.tight_layout()
                plt.savefig(vis_path)
                plt.close()
        else:
            features = hog(aug_image, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), visualize=False)

        result.append((features, person_name))

    return result


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Data augmentation test")
    parser.add_argument("--image", default='Cuong_000.jpg', help="Path to a test image")
    parser.add_argument("--output_dir", default="augmentation_examples", help="Output directory for examples")
    args = parser.parse_args()

    save_augmentation_examples(args.image, args.output_dir)
