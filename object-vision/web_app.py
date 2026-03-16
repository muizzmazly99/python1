import os
import cv2
import time
import pyttsx3
import face_recognition
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

app = FastAPI()
templates = Jinja2Templates(directory="templates")

engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


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

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

last_spoken_object = ""
last_spoken_face = ""
last_unknown_face_time = 0
UNKNOWN_FACE_COOLDOWN = 5  # seconds

object_stats = {}
recognized_faces = []
event_log = []
last_detected = "None"


def add_event(message: str):
    global event_log
    timestamp = time.strftime("%H:%M:%S")
    entry = f"{timestamp} - {message}"

    if len(event_log) == 0 or event_log[0] != entry:
        event_log.insert(0, entry)
        event_log = event_log[:10]


def generate_frames():
    global last_spoken_object, last_spoken_face, last_unknown_face_time
    global object_stats, last_detected, recognized_faces

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        boxes = results[0].boxes

        object_counts = {}
        object_stats = {}
        current_faces = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                object_counts[label] = object_counts.get(label, 0) + 1
                object_stats[label] = object_stats.get(label, 0) + 1
                last_detected = label

            first_label = list(object_counts.keys())[0]
            if first_label != last_spoken_object:
                add_event(f"Object detected: {first_label}")
                # speak(f"{first_label} detected")
                last_spoken_object = first_label
        else:
            last_detected = "None"

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

            current_faces.append(name)

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
            add_event(f"Known face detected: {name}")
            # speak(f"{name} detected")
            last_spoken_face = name
        elif name == "Unknown":
            current_time = time.time()
            global last_unknown_face_time

            if current_time - last_unknown_face_time >= UNKNOWN_FACE_COOLDOWN:
                add_event("Unknown face detected")
                last_unknown_face_time = current_time

        recognized_faces = list(dict.fromkeys(current_faces))

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/stats")
def get_stats():
    return {
        "objects": object_stats,
        "last_detected": last_detected,
        "faces": recognized_faces,
        "events": event_log
    }
