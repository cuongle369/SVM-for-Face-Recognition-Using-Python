import cv2
import joblib
from skimage.feature import hog
import argparse

def extract_hog_features(image):
    features = hog(image, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=False)
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svm_model", default="svm_model.joblib", help="Path to the trained SVM model")
    parser.add_argument("--label_encoder", default="label_encoder.joblib", help="Path to the label encoder")
    args = parser.parse_args()

    svm = joblib.load(args.svm_model)
    label_encoder = joblib.load(args.label_encoder)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (160,160))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            features = extract_hog_features(face_gray)
            prediction = svm.predict([features])
            person_name = label_encoder.inverse_transform(prediction)[0]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()