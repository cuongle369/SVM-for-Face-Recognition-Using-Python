import cv2
import os
import argparse
import shutil

# Initialize Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Function to extract frames from a video
def extract_frames(video_path, output_dir, interval=1):
    """
    Extract frames from a video and save them to the output directory.
    - video_path: Path to the video file
    - output_dir: Directory to save extracted frames
    - interval: Interval between extracted frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return 0

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")
    return saved_count


# Function to detect and crop faces
def detect_and_crop_faces(image_path, output_dir, person_name, face_counter, min_face_size=(100, 100)):
    """
    Detect faces in an image, crop, preprocess, and save them.
    - image_path: Path to the image
    - output_dir: Directory to save detected faces
    - person_name: Person's name for naming files
    - face_counter: Global face counter
    - min_face_size: Minimum face size
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return 0, face_counter

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=min_face_size)

    face_count = 0
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (160, 160))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        face_normalized = cv2.equalizeHist(face_gray)

        face_path = os.path.join(output_dir, f"{person_name}_{face_counter:03d}.jpg")
        cv2.imwrite(face_path, face_normalized)
        face_count += 1
        face_counter += 1

    return face_count, face_counter


# Function to process all data for a person
def process_person(person_name, video_dir, output_base_dir, interval=1):
    """
    Process data for a person: extract frames and detect faces.
    - person_name: Name of the person
    - video_dir: Directory containing videos
    - output_base_dir: Main directory to store processed data
    - interval: Frame extraction interval
    """
    person_dir = os.path.join(output_base_dir, person_name)
    temp_dir = os.path.join(person_dir, "temp_frames")

    # Create directories if they do not exist
    os.makedirs(person_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Find all videos in the directory
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
    if not video_files:
        print(f"No videos found in {video_dir}")
        return

    total_frames = 0
    total_faces = 0
    face_counter = 0  # Global face counter

    # Extract frames from each video
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        frames_extracted = extract_frames(video_path, temp_dir, interval)
        total_frames += frames_extracted

    # Detect and process faces from extracted frames
    frame_files = [f for f in os.listdir(temp_dir) if f.startswith('frame_')]
    for frame_file in frame_files:
        frame_path = os.path.join(temp_dir, frame_file)
        faces_detected, face_counter = detect_and_crop_faces(frame_path, person_dir, person_name, face_counter)
        total_faces += faces_detected
        os.remove(frame_path)  # Remove frame after processing

    # Remove temporary directory
    shutil.rmtree(temp_dir)
    print(f"Processing complete for {person_name}: {total_frames} frames, {total_faces} faces")


# Main function
def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Collect and prepare face data from videos.")
    parser.add_argument("--input_dir", default="input_videos", help="Directory containing input videos")
    parser.add_argument("--output_dir", default="face_data", help="Directory to save processed data")
    parser.add_argument("--interval", type=int, default=5, help="Frame extraction interval")
    args = parser.parse_args()

    # List of 4 people in the group
    people = ["Cuong", "Ly", "MinhSon", "PhienLuu"]

    # Create output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting data processing...")

    # Process each person
    for person in people:
        video_dir = os.path.join(args.input_dir, person)
        if not os.path.exists(video_dir):
            print(f"Directory {video_dir} does not exist, skipping {person}")
            continue
        process_person(person, video_dir, args.output_dir, args.interval)

    print("Data collection and preparation process complete!")


if __name__ == "__main__":
    main()
