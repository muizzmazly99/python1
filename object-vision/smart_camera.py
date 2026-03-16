import os
import cv2
import pyttsx3
import face_recognition
from ultralytics import YOLO

# -------------------------
# Text-to-speech
# -------------------------
engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


# -------------------------
# Load known faces
# -------------------------
KNOWN_FACES_DIR = "known_faces"
known_face_encodings = []
known_face_names = []

if os.path.exists(KNOWN_FACES_DIR):
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
            else:
                print(f"No face found in {filename}")
else:
    print("Warning: known_faces folder not found.")

# -------------------------
# Main app
# -------------------------


def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit.")

    last_spoken_object = ""
    last_spoken_face = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # -------------------------
        # YOLO object detection
        # -------------------------
        results = model(frame)
        annotated_frame = results[0].plot()
        boxes = results[0].boxes
        object_counts = {}

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                object_counts[label] = object_counts.get(label, 0) + 1

            first_label = list(object_counts.keys())[0]
            if first_label != last_spoken_object:
                speak(f"{first_label} detected")
                last_spoken_object = first_label

        # Draw object counts
        y = 30
        for label, count in object_counts.items():
            text = f"{label}: {count}"
            cv2.putText(
                annotated_frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            y += 30

        # -------------------------
        # Face recognition
        # -------------------------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = "Unknown"

            if len(known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding)
                best_match_index = face_distances.argmin()

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            # Draw face box
            cv2.rectangle(annotated_frame, (left, top),
                          (right, bottom), (255, 0, 0), 2)
            cv2.rectangle(annotated_frame, (left, bottom - 35),
                          (right, bottom), (255, 0, 0), cv2.FILLED)
            cv2.putText(
                annotated_frame,
                name,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            if name != "Unknown" and name != last_spoken_face:
                speak(f"{name} detected")
                last_spoken_face = name

        cv2.imshow("Smart Camera", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
