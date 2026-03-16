import cv2
from ultralytics import YOLO
import pyttsx3

engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit.")

    last_spoken = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        boxes = results[0].boxes
        object_counts = {}

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                object_counts[label] = object_counts.get(label, 0) + 1

            # Speak only the first detected object if it changed
            first_label = list(object_counts.keys())[0]
            if first_label != last_spoken:
                speak(f"{first_label} detected")
                last_spoken = first_label

        # Draw object counts on top-left
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

        cv2.imshow("Object Detection + Counter", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
