import os
import cv2
import numpy as np
from skimage.feature import hog
import pickle
import argparse
import matplotlib.pyplot as plt


def extract_hog_features(image, visualize=True):
    """
    Trích xuất đặc trưng HOG từ ảnh và tùy chọn trả về hình ảnh trực quan hóa

    Parameters:
    - image: ảnh đầu vào
    - visualize: nếu True, trả về cả đặc trưng và ảnh trực quan hóa

    Returns:
    - features: vector đặc trưng HOG
    - hog_image: ảnh trực quan hóa HOG (nếu visualize=True)
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
    Lưu ảnh gốc và ảnh trực quan hóa HOG cạnh nhau

    Parameters:
    - original_image: ảnh gốc
    - hog_image: ảnh trực quan hóa HOG
    - output_path: đường dẫn để lưu ảnh kết quả
    """
    # Tạo figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Hiển thị ảnh gốc
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Hiển thị ảnh HOG
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
    args = parser.parse_args()

    # Tạo thư mục trực quan hóa nếu chưa tồn tại
    os.makedirs(args.vis_dir, exist_ok=True)

    features = []
    labels = []

    for person in os.listdir(args.base_dir):
        person_dir = os.path.join(args.base_dir, person)
        if os.path.isdir(person_dir):
            # Tạo thư mục trực quan hóa cho từng người
            person_vis_dir = os.path.join(args.vis_dir, person)
            os.makedirs(person_vis_dir, exist_ok=True)

            for image_file in os.listdir(person_dir):
                if image_file.endswith(".jpg"):
                    image_path = os.path.join(person_dir, image_file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    if image is not None:
                        # Trích xuất đặc trưng HOG với visualize=True
                        feat, hog_image = extract_hog_features(image, visualize=True)

                        # Lưu đặc trưng và nhãn
                        features.append(feat)
                        labels.append(person)

                        # Lưu ảnh trực quan hóa
                        vis_filename = os.path.splitext(image_file)[0] + "_hog.png"
                        vis_path = os.path.join(person_vis_dir, vis_filename)
                        save_visualization(image, hog_image, vis_path)

                        print(f"Processed and visualized: {image_file}")

    # Lưu đặc trưng và nhãn vào file pickle
    with open(args.output_file, 'wb') as f:
        pickle.dump((features, labels), f)

    print(f"Features extracted and saved to {args.output_file}")
    print(f"Visualizations saved to {args.vis_dir}")


if __name__ == "__main__":
    main()