import cv2
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import easyocr
import threading

# Constants
FPS = 30  # Frame rate video
DISTANCE_PER_PIXEL = 0.006  # Estimasi jarak tiap piksel dalam meter (harus dikalibrasi)
SPEED_THRESHOLD = 20  # Ambang batas kecepatan dalam km/jam
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for object detection

# Inisialisasi model YOLO
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'])  # Inisialisasi EasyOCR untuk membaca plat nomor

# Mapping class ID ke label kendaraan (hanya mobil dan motor)
COCO_CLASSES = {
    2: "Mobil",
    3: "Motor",
    5: "Bus",
    7: "Truk",
    0: "Orang",
    1: "Sepeda",
    # Bisa ditambahkan objek lainnya sesuai dataset COCO
}

def match_objects(prev_boxes, new_boxes, threshold=0.5):
    """
    Fungsi untuk mencocokkan objek berdasarkan IoU
    """
    matched_objects = {}
    for new_id, new_box in enumerate(new_boxes):
        if isinstance(new_box, tuple) and len(new_box) == 4:
            matched_objects[new_id] = new_box
    return matched_objects

def calculate_speed(prev_center, curr_center, fps, distance_per_pixel):
    """
    Fungsi untuk menghitung kecepatan kendaraan
    """
    distance_px = np.sqrt((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2)
    distance_m = distance_px * distance_per_pixel  # Konversi ke meter
    speed_mps = distance_m * fps  # Kecepatan dalam meter per detik
    speed_kmh = speed_mps * 3.6  # Konversi ke km/jam
    return speed_kmh

def detect_license_plate(image, bbox):
    """
    Fungsi untuk mendeteksi plat nomor dengan peningkatan akurasi
    """
    if isinstance(bbox, tuple) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        plate_region = image[y1:y2, x1:x2]
        gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray_plate)
        if results:
            return results[0][-2]  # Mengembalikan teks plat nomor
    return "Unknown"

def process_camera(camera_index, window_name):
    """
    Fungsi untuk memproses video dari satu kamera
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {camera_index}")
        return

    prev_positions = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        new_positions = {}
        
        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                if conf > CONFIDENCE_THRESHOLD and int(cls) in COCO_CLASSES:
                    x1, y1, x2, y2 = map(int, box)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    new_positions[len(new_positions)] = (x1, y1, x2, y2)
                    
                    # Identifikasi jenis kendaraan
                    vehicle_type = COCO_CLASSES[int(cls)]
                    
                    # Deteksi plat nomor
                    plate_number = detect_license_plate(frame, (x1, y2 - 30, x2, y2))
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{vehicle_type} - {plate_number}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        matched_objects = match_objects(prev_positions, new_positions)
        
        for object_id, bbox in matched_objects.items():
            if isinstance(bbox, tuple) and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                if object_id in prev_positions:
                    prev_center = ((prev_positions[object_id][0] + prev_positions[object_id][2]) // 2,
                                   (prev_positions[object_id][1] + prev_positions[object_id][3]) // 2)
                    speed = calculate_speed(prev_center, center, FPS, DISTANCE_PER_PIXEL)
                    
                    if speed > SPEED_THRESHOLD:
                        cv2.putText(frame, f"{speed:.2f} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, f"{speed:.2f} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        prev_positions = matched_objects
        
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Jalankan dua kamera secara paralel
thread1 = threading.Thread(target=process_camera, args=(0, "Camera 1"))
thread2 = threading.Thread(target=process_camera, args=(1, "Camera 2"))

thread1.start()
thread2.start()

thread1.join()
thread2.join()
