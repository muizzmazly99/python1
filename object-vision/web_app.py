import os
import cv2
import time
import pyttsx3
import face_recognition
from fastapi import FastAPI, Request, HTTPException
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

camera_sources = {
    "webcam": 0,
    "mobile": "http://192.168.1.20:8080/video"
}

active_camera = "webcam"
cap = None

last_spoken_object = ""
last_spoken_face = ""
last_unknown_face_time = 0
UNKNOWN_FACE_COOLDOWN = 5

frame_count = 0
last_face_names = []
last_face_data = []  # [(top, right, bottom, left, name)]

object_stats = {}
recognized_faces = []
event_log = []
last_detected = "None"

active_track_ids = set()
seen_track_ids = set()
tracked_objects = {}


def add_event(message: str):
    global event_log
    timestamp = time.strftime("%H:%M:%S")
    entry = f"{timestamp} - {message}"

    if len(event_log) == 0 or event_log[0] != entry:
        event_log.insert(0, entry)
        event_log = event_log[:10]


def set_camera_source(source_name: str):
    global cap, active_camera, frame_count, last_face_names, last_face_data
    global active_track_ids, seen_track_ids, tracked_objects

    if source_name not in camera_sources:
        raise ValueError(f"Unknown camera source: {source_name}")

    new_source = camera_sources[source_name]
    new_cap = cv2.VideoCapture(new_source)

    if not new_cap.isOpened():
        raise RuntimeError(f"Failed to open camera source: {source_name}")

    if cap is not None:
        cap.release()

    cap = new_cap
    active_camera = source_name
    frame_count = 0
    last_face_names = []
    last_face_data = []
    active_track_ids = set()
    seen_track_ids = set()
    tracked_objects = {}

    add_event(f"Camera switched to: {source_name}")


try:
    set_camera_source(active_camera)
except RuntimeError:
    try:
        set_camera_source("webcam")
    except RuntimeError:
        cap = None
        add_event("No camera source available at startup")


def generate_frames():
    global last_spoken_object, last_spoken_face, last_unknown_face_time
    global object_stats, last_detected, recognized_faces
    global frame_count, last_face_names, last_face_data
    global active_track_ids, seen_track_ids, tracked_objects
    global cap

    while True:
        if cap is None:
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        frame_count += 1
        frame = cv2.resize(frame, (640, 480))

        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        boxes = results[0].boxes

        active_ids_this_frame = set()
        current_tracked_objects = {}

        object_counts = {}
        object_stats = {}

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                track_id = int(box.id[0]) if box.id is not None else -1

                object_counts[label] = object_counts.get(label, 0) + 1
                object_stats[label] = object_stats.get(label, 0) + 1
                last_detected = label

                if track_id != -1:
                    active_ids_this_frame.add(track_id)
                    current_tracked_objects[track_id] = label

                    if track_id not in seen_track_ids:
                        seen_track_ids.add(track_id)
                        add_event(f"{label.capitalize()} #{track_id} detected")

        else:
            last_detected = "None"

        lost_ids = active_track_ids - active_ids_this_frame
        for lost_id in lost_ids:
            old_label = tracked_objects.get(lost_id, "Object")
            add_event(f"{old_label.capitalize()} #{lost_id} left view")

        active_track_ids = active_ids_this_frame
        tracked_objects = current_tracked_objects

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

        # Run face recognition every 5 frames
        if frame_count % 5 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_frame, face_locations)

            current_faces = []
            current_face_data = []

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding)
                name = "Unknown"

                if len(known_face_encodings) > 0:
                    face_distances = face_recognition.face_distance(
                        known_face_encodings,
                        face_encoding
                    )
                    best_match_index = face_distances.argmin()

                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                current_faces.append(name)
                current_face_data.append((top, right, bottom, left, name))

                if name != "Unknown" and name != last_spoken_face:
                    add_event(f"Known face detected: {name}")
                    last_spoken_face = name
                elif name == "Unknown":
                    current_time = time.time()
                    if current_time - last_unknown_face_time >= UNKNOWN_FACE_COOLDOWN:
                        add_event("Unknown face detected")
                        last_unknown_face_time = current_time

            last_face_names = current_faces
            last_face_data = current_face_data

        # Draw cached face results every frame
        for top, right, bottom, left, name in last_face_data:
            cv2.rectangle(annotated_frame, (left, top),
                          (right, bottom), (255, 0, 0), 2)
            cv2.rectangle(
                annotated_frame,
                (left, bottom - 35),
                (right, bottom),
                (255, 0, 0),
                cv2.FILLED
            )
            cv2.putText(
                annotated_frame,
                name,
                (left + 6, bottom - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        recognized_faces = list(dict.fromkeys(last_face_names))

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            continue

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
        "events": event_log,
        "active_camera": active_camera,
        "active_tracks": list(active_track_ids),
        "total_unique_tracks": len(seen_track_ids)
    }


@app.get("/camera_sources")
def get_camera_sources():
    return {
        "available_sources": list(camera_sources.keys()),
        "active_camera": active_camera
    }


@app.post("/switch_camera/{source_name}")
def switch_camera(source_name: str):
    try:
        set_camera_source(source_name)
        return {
            "message": f"Switched to {source_name}",
            "active_camera": active_camera
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
