import cv2
import os
import argparse
import shutil

# Khởi tạo bộ phân loại Haar Cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Hàm trích xuất khung hình từ video
def extract_frames(video_path, output_dir, interval=1):
    """
    Trích xuất khung hình từ video và lưu vào thư mục đầu ra.
    - video_path: Đường dẫn đến file video
    - output_dir: Thư mục lưu khung hình
    - interval: Khoảng cách giữa các khung được trích xuất
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
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
    print(f"Đã trích xuất {saved_count} khung hình từ {video_path}")
    return saved_count


# Hàm phát hiện và cắt khuôn mặt
def detect_and_crop_faces(image_path, output_dir, person_name, face_counter, min_face_size=(50, 50)):
    """
    Phát hiện khuôn mặt từ ảnh, cắt và tiền xử lý, sau đó lưu lại.
    - image_path: Đường dẫn đến ảnh
    - output_dir: Thư mục lưu khuôn mặt
    - person_name: Tên người để đặt tên file
    - face_counter: Bộ đếm khuôn mặt toàn cục
    - min_face_size: Kích thước tối thiểu của khuôn mặt
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
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


# Hàm xử lý toàn bộ dữ liệu cho một người
def process_person(person_name, video_dir, output_base_dir, interval=1):
    """
    Xử lý dữ liệu cho một người: trích xuất khung hình và phát hiện khuôn mặt.
    - person_name: Tên người
    - video_dir: Thư mục chứa video
    - output_base_dir: Thư mục chính để lưu dữ liệu
    - interval: Khoảng cách trích xuất khung
    """
    person_dir = os.path.join(output_base_dir, person_name)
    temp_dir = os.path.join(person_dir, "temp_frames")

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(person_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Tìm tất cả video trong thư mục
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
    if not video_files:
        print(f"Không tìm thấy video nào trong {video_dir}")
        return

    total_frames = 0
    total_faces = 0
    face_counter = 0  # Bộ đếm khuôn mặt toàn cục

    # Trích xuất khung hình từ mỗi video
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        frames_extracted = extract_frames(video_path, temp_dir, interval)
        total_frames += frames_extracted

    # Phát hiện và xử lý khuôn mặt từ các khung hình
    frame_files = [f for f in os.listdir(temp_dir) if f.startswith('frame_')]
    for frame_file in frame_files:
        frame_path = os.path.join(temp_dir, frame_file)
        faces_detected, face_counter = detect_and_crop_faces(frame_path, person_dir, person_name, face_counter)
        total_faces += faces_detected
        os.remove(frame_path)  # Xóa khung hình sau khi xử lý

    # Xóa thư mục tạm
    shutil.rmtree(temp_dir)
    print(f"Hoàn tất xử lý cho {person_name}: {total_frames} khung hình, {total_faces} khuôn mặt")


# Hàm chính
def main():
    # Thiết lập đối số dòng lệnh
    parser = argparse.ArgumentParser(description="Thu thập và chuẩn bị dữ liệu khuôn mặt từ video.")
    parser.add_argument("--input_dir", default="input_videos", help="Thư mục chứa video đầu vào")
    parser.add_argument("--output_dir", default="face_data", help="Thư mục lưu dữ liệu đã xử lý")
    parser.add_argument("--interval", type=int, default=5, help="Khoảng cách trích xuất khung hình")
    args = parser.parse_args()

    # Danh sách 4 người trong nhóm
    people = ["Cuong", "Ly", "MinhSon", "PhienLuu"]

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(args.output_dir, exist_ok=True)

    print("Bắt đầu xử lý dữ liệu...")

    # Xử lý từng người
    for person in people:
        video_dir = os.path.join(args.input_dir, person)
        if not os.path.exists(video_dir):
            print(f"Thư mục {video_dir} không tồn tại, bỏ qua {person}")
            continue
        process_person(person, video_dir, args.output_dir, args.interval)

    print("Quá trình thu thập và chuẩn bị dữ liệu hoàn tất!")


if __name__ == "__main__":
    main()