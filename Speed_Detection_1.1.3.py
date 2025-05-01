import cv2
import time
import threading
from ultralytics import YOLO
import numpy as np
import pytesseract
from datetime import datetime

# Constants
FPS = 30
DISTANCE_PER_PIXEL = 0.00625  # ~1m = 160px (4m height)
SPEED_THRESHOLD = 20  # km/h
CONFIDENCE_THRESHOLD = 0.5
SWITCH_INTERVAL = 5  # seconds

# Load YOLO model
model = YOLO("yolov8n.pt")  # Replace with .onnx version if needed
COCO_CLASSES = {2: "Mobil", 3: "Motor"}

# OCR function (runs in a separate thread)
def run_ocr_async(frame, bbox, cam_index, speed_kmh):
    def task():
        x1, y1, x2, y2 = bbox
        plate_region = frame[y2 - 30:y2, x1:x2]
        if plate_region.size == 0:
            log_result("Unknown", speed_kmh, cam_index)
            return

        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        plate_text = pytesseract.image_to_string(gray, config='--psm 7').strip()
        log_result(plate_text, speed_kmh, cam_index)
    threading.Thread(target=task).start()

# Log speed violations
def log_result(plate, speed, cam_index):
    with open("speed_log.txt", "a") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] Cam {cam_index} | Speed: {speed:.2f} km/h | Plate: {plate}\n")
    print(f"[LOG] Cam {cam_index} | {speed:.2f} km/h | Plate: {plate}")

# Speed calculation
def calculate_speed(prev, curr):
    dist_px = np.linalg.norm(np.array(curr) - np.array(prev))
    speed = dist_px * DISTANCE_PER_PIXEL * FPS * 3.6
    return speed

# Process camera
def process_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Kamera {camera_index} tidak bisa dibuka.")
        return

    prev_positions = {}
    start = time.time()

    while time.time() - start < SWITCH_INTERVAL:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        new_positions = {}

        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                if int(cls) in COCO_CLASSES and conf > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    label = COCO_CLASSES[int(cls)]
                    object_id = len(new_positions)

                    new_positions[object_id] = {
                        "bbox": (x1, y1, x2, y2),
                        "center": center,
                        "label": label
                    }

                    if object_id in prev_positions:
                        prev_center = prev_positions[object_id]["center"]
                        speed = calculate_speed(prev_center, center)
                        color = (0, 0, 255) if speed > SPEED_THRESHOLD else (255, 255, 255)
                        cv2.putText(frame, f"{speed:.1f} km/h", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        if speed > SPEED_THRESHOLD:
                            run_ocr_async(frame.copy(), (x1, y1, x2, y2), camera_index, speed)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        prev_positions = new_positions
        cv2.imshow(f"Kamera {camera_index}", frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main loop: alternating cameras
while True:
    process_camera(0)
    process_camera(1)
