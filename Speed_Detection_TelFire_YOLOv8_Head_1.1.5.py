import cv2
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import easyocr
import threading
import asyncio
import telebot
import time
import os
from PIL import Image
import io
import logging
import firebase_admin
from firebase_admin import credentials, db, storage
import base64
import uuid
import torch

# Inisialisasi Firebase
cred = credentials.Certificate("speed-detection-109da-firebase-adminsdk-fbsvc-d9120017d2.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://speed-detection-109da-default-rtdb.asia-southeast1.firebasedatabase.app/',
})

# Initialize Firebase Realtime Database references
violations_ref = db.reference('violations')
system_stats_ref = db.reference('system_stats')
cameras_ref = db.reference('cameras')
daily_stats_ref = db.reference('daily_stats')

# Telegram Bot Setup
BOT_TOKEN = "7633151627:AAFowoEJTa9In8nYpHccAi9fSBP92Vw5lik"
CHAT_ID = "6451128792"
bot = telebot.TeleBot(BOT_TOKEN)

# Konstanta
FPS = 25
DISTANCE_PER_PIXEL = 0.007
SPEED_THRESHOLD = 30
CONFIDENCE_THRESHOLD = 0.5
NOTIFICATION_COOLDOWN = 10
ACTIVITY_LOG_INTERVAL = 60
MIN_DISPLACEMENT_PX = 5  # Minimum movement in pixels to consider for speed calculation
MIN_MOVEMENT_FRAMES = 3  # Minimum consecutive frames with movement to consider vehicle moving

# Set up logging - MODIFIED FIX
logger = logging.getLogger("SpeedDetectionSystem")
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler with UTF-8 encoding
file_handler = logging.FileHandler("speed_detection_activity.log", encoding='utf-8')
file_handler.setFormatter(formatter)

# Stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Inisialisasi model dan OCR
model = YOLO("yolov8n.pt").to('cpu')  # Force CPU usage
torch.set_num_threads(4)  # Limit CPU threads
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./ocr_models',
                      download_enabled=True, detector=True, recognizer=True)

# Kelas objek yang ingin dideteksi
COCO_CLASSES = {
    0: "Orang",
    2: "Mobil",
    3: "Motor"
}

# Tracking last notification time for each license plate
last_notification = {}

# Global statistics dictionary
system_stats = {
    "start_time": None,
    "cameras": {},
    "total_vehicles_detected": 0,
    "total_violations": 0,
    "last_activity_report": 0
}

def save_violation_to_firebase(camera_name, vehicle_type, plate_number, speed):
    """Save violation metadata to Firebase (without images)"""
    try:
        violation_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        violation_data = {
            'id': violation_id,
            'camera_name': camera_name,
            'vehicle_type': vehicle_type,
            'license_plate': plate_number,
            'speed_kmh': round(speed, 2),
            'speed_threshold': SPEED_THRESHOLD,
            'capture_time': current_time.isoformat(),
            'timestamp': int(time.time()),
            'location': {
                'camera_id': camera_name.lower().replace(' ', '_'),
                'coordinates': None
            },
            'notified': True,
            'created_at': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'telegram_sent': True
        }
        
        # Save violation to Firebase
        violations_ref.child(violation_id).set(violation_data)
        
        # Update daily statistics
        today = current_time.strftime('%Y-%m-%d')
        daily_stats_ref.child(today).transaction(lambda current_value: 
            (current_value or 0) + 1)
        
        # Update camera-specific statistics
        camera_stats_ref = cameras_ref.child(camera_name.lower().replace(" ", "_"))
        camera_stats_ref.child('total_violations').transaction(lambda current_value: 
            (current_value or 0) + 1)
        camera_stats_ref.child('last_violation').set(current_time.isoformat())
        
        log_activity(f"Violation metadata saved to Firebase with ID: {violation_id}")
        return violation_id
        
    except Exception as e:
        log_activity(f"Error saving violation metadata to Firebase: {str(e)}")
        return None

def update_system_stats_firebase():
    """Update system statistics in Firebase"""
    try:
        current_time = datetime.now()
        uptime = current_time - system_stats["start_time"] if system_stats["start_time"] else None
        
        firebase_stats = {
            'start_time': system_stats["start_time"].isoformat() if system_stats["start_time"] else None,
            'uptime_seconds': int(uptime.total_seconds()) if uptime else 0,
            'total_vehicles_detected': system_stats["total_vehicles_detected"],
            'total_violations': system_stats["total_violations"],
            'last_update': current_time.isoformat(),
            'speed_threshold': SPEED_THRESHOLD,
            'system_status': 'active'
        }
        
        system_stats_ref.set(firebase_stats)
        
        # Update camera statuses
        for cam_name, stats in system_stats["cameras"].items():
            camera_data = {
                'name': cam_name,
                'active': stats["active"],
                'vehicles_detected': stats["vehicles_detected"],
                'violations': stats["violations"],
                'last_frame_time': stats["last_frame_time"],
                'last_update': current_time.isoformat()
            }
            cameras_ref.child(cam_name.lower().replace(' ', '_')).set(camera_data)
            
    except Exception as e:
        log_activity(f"Error updating Firebase stats: {str(e)}")

def log_activity(message, send_to_telegram=False):
    """Log activity to file and optionally send to Telegram"""
    logger.info(message)
    if send_to_telegram and CHAT_ID:
        bot.send_message(chat_id=CHAT_ID, text=message)

def match_objects(prev_boxes, new_boxes):
    matched_objects = {}
    for new_id, data in new_boxes.items():
        if isinstance(data, dict) and "bbox" in data:
            matched_objects[new_id] = data
    return matched_objects

def calculate_speed(prev_center, curr_center, time_delta, distance_per_pixel):
    """Calculate speed using actual time difference"""
    if time_delta <= 0:
        return 0.0
        
    distance_px = np.sqrt((curr_center[0] - prev_center[0]) ** 2 + 
                         (curr_center[1] - prev_center[1]) ** 2)
    distance_m = distance_px * distance_per_pixel
    speed_mps = distance_m / time_delta
    speed_kmh = speed_mps * 3.6
    return speed_kmh

def detect_license_plate(image, bbox):
    """Optimized license plate detection for Indonesian plates (black on white)"""
    if isinstance(bbox, tuple) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        
        # Focus on the bottom 30% of the vehicle where plates are typically located
        plate_region_height = int((y2 - y1) * 0.3)
        plate_y1 = max(0, y2 - plate_region_height)
        plate_y2 = min(image.shape[0], y2 + 5)
        plate_x1 = max(0, x1)
        plate_x2 = min(image.shape[1], x2)
        
        plate_region = image[plate_y1:plate_y2, plate_x1:plate_x2]
        
        if plate_region.size == 0:
            return "Unknown"
            
        # Indonesian Plate-specific processing
        try:
            # Convert to grayscale and enhance contrast
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply sharpening filter
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Binarization optimized for black-on-white plates
            _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if needed (should be white background with black text)
            if np.mean(thresh) < 127:
                thresh = cv2.bitwise_not(thresh)
                
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Scale up for low-resolution plates
            scaled = cv2.resize(cleaned, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Try OCR with Indonesian plate constraints
            results = reader.readtext(
                scaled,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
                min_size=10,
                text_threshold=0.3,
                width_ths=0.5,
                decoder='greedy'
            )
            
            # Filter and validate results
            valid_results = []
            for res in results:
                text = res[1].upper().replace(" ", "")
                if 3 <= len(text) <= 10:
                    letter_count = sum(c.isalpha() for c in text)
                    digit_count = sum(c.isdigit() for c in text)
                    if letter_count >= 1 and digit_count >= 2:
                        valid_results.append((text, res[2]))
            
            if valid_results:
                valid_results.sort(key=lambda x: x[1], reverse=True)
                return valid_results[0][0]
                
        except Exception as e:
            log_activity(f"Plate processing error: {str(e)}")
    
    return "Unknown"

def send_telegram_notification(camera_name, vehicle_type, plate_number, speed, frame):
    """Enhanced notification with multiple views of the vehicle"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    violation_id = save_violation_to_firebase(camera_name, vehicle_type, plate_number, speed)
    
    # Create a directory for violation images
    violation_dir = "violation_images"
    if not os.path.exists(violation_dir):
        os.makedirs(violation_dir)
    
    # Base filename for this violation
    base_filename = f"{violation_dir}/violation_{int(time.time())}"
    
    # Save the main frame
    main_image_path = f"{base_filename}_main.jpg"
    cv2.imwrite(main_image_path, frame)
    
    # Prepare message
    message = f"⚠️ PERINGATAN KECEPATAN BERLEBIH! ⚠️\n\n"
    message += f"📹 Kamera: {camera_name}\n"
    message += f"🚗 Tipe Kendaraan: {vehicle_type}\n"
    message += f"🔢 Plat Nomor: {plate_number}\n"
    message += f"⚡ Kecepatan: {speed:.2f} km/jam\n"
    message += f"⏰ Waktu: {current_time}\n"
    message += f"🆔 ID Pelanggaran: {violation_id}\n"
    message += f"\nKendaraan melebihi batas kecepatan {SPEED_THRESHOLD} km/jam!"
    
    # Extract bounding box coordinates
    for r in model(frame):
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            class_id = int(cls)
            if conf > CONFIDENCE_THRESHOLD and class_id in COCO_CLASSES:
                if COCO_CLASSES[class_id] == vehicle_type:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Extract zoomed-in view of the vehicle
                    try:
                        # Expand the crop region
                        height, width = frame.shape[:2]
                        crop_y1 = max(0, y1 - 10)
                        crop_y2 = min(height, y2 + 10)
                        crop_x1 = max(0, x1 - 10)
                        crop_x2 = min(width, x2 + 10)
                        
                        vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        if vehicle_crop.size > 0:
                            zoomed_image_path = f"{base_filename}_zoomed.jpg"
                            cv2.imwrite(zoomed_image_path, vehicle_crop)
                            
                            # Try to extract the license plate region
                            plate_y1 = int(y2 - (y2-y1) * 0.3)
                            plate_y2 = y2
                            plate_crop = frame[plate_y1:plate_y2, x1:x2]
                            
                            if plate_crop.size > 0:
                                # Enhance the license plate image
                                gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                                enhanced_plate = clahe.apply(gray_plate)
                                
                                plate_image_path = f"{base_filename}_plate.jpg"
                                cv2.imwrite(plate_image_path, enhanced_plate)
                    except Exception as e:
                        log_activity(f"Error creating additional screenshots: {str(e)}")
    
    # Send main image with caption
    with open(main_image_path, 'rb') as photo:
        bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=message)
    
    # Send zoomed image if available
    zoomed_image_path = f"{base_filename}_zoomed.jpg"
    if os.path.exists(zoomed_image_path):
        caption = f"Gambar Zoom Kendaraan - {vehicle_type} dengan plat {plate_number}"
        with open(zoomed_image_path, 'rb') as photo:
            bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=caption)
    
    # Send plate image if available
    plate_image_path = f"{base_filename}_plate.jpg"
    if os.path.exists(plate_image_path):
        # Run OCR again on the enhanced plate image
        enhanced_plate = cv2.imread(plate_image_path, cv2.IMREAD_GRAYSCALE)
        if enhanced_plate is not None:
            plate_results = reader.readtext(
                enhanced_plate,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
                min_size=10,
                text_threshold=0.3
            )
            enhanced_plate_number = "Unknown"
            
            if plate_results:
                # Filter and validate results
                valid_results = []
                for res in plate_results:
                    text = res[1].upper().replace(" ", "")
                    if 3 <= len(text) <= 10:
                        letter_count = sum(c.isalpha() for c in text)
                        digit_count = sum(c.isdigit() for c in text)
                        if letter_count >= 1 and digit_count >= 2:
                            valid_results.append((text, res[2]))
                
                if valid_results:
                    valid_results.sort(key=lambda x: x[1], reverse=True)
                    enhanced_plate_number = valid_results[0][0]
            
            caption = f"Plat Nomor: {enhanced_plate_number if len(enhanced_plate_number) >= 2 else plate_number}"
            with open(plate_image_path, 'rb') as photo:
                bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=caption)
    
    # Update stats
    system_stats["total_violations"] += 1
    log_activity(f"Notification sent for {plate_number} at {speed:.2f} km/h on {camera_name}")

def send_activity_report():
    """Send a comprehensive activity report to Telegram"""
    if not CHAT_ID:
        return
    
    current_time = datetime.now()
    uptime = current_time - system_stats["start_time"]
    uptime_str = str(uptime).split('.')[0]
    
    # Create activity report
    message = f"📊 SISTEM PEMANTAUAN KECEPATAN - LAPORAN AKTIVITAS 📊\n\n"
    message += f"⏱️ Waktu Operasi: {uptime_str}\n"
    message += f"🚗 Total Kendaraan Terdeteksi: {system_stats['total_vehicles_detected']}\n"
    message += f"🚨 Total Pelanggaran: {system_stats['total_violations']}\n\n"
    
    # Add camera-specific stats
    message += "📷 STATUS KAMERA:\n"
    for cam_name, stats in system_stats["cameras"].items():
        status = "✅ Aktif" if stats["active"] else "❌ Tidak Aktif"
        message += f"- {cam_name}: {status}\n"
        message += f"  Kendaraan: {stats['vehicles_detected']}\n"
        message += f"  Pelanggaran: {stats['violations']}\n"
    
    message += f"\n⚙️ Batas Kecepatan: {SPEED_THRESHOLD} km/jam"
    
    bot.send_message(chat_id=CHAT_ID, text=message)
    system_stats["last_activity_report"] = time.time()
    log_activity("Activity report sent to Telegram")

# Telegram bot commands
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Selamat datang di Sistem Pemantauan Kecepatan Kendaraan! Bot siap memantau.")
    global CHAT_ID
    CHAT_ID = message.chat.id
    bot.send_message(chat_id=CHAT_ID, text=f"Chat ID Anda adalah: {CHAT_ID}")
    log_activity(f"Bot started by user with chat ID: {CHAT_ID}")

@bot.message_handler(commands=['status'])
def send_status(message):
    if system_stats["start_time"]:
        send_activity_report()
    else:
        bot.reply_to(message, "Sistem belum diinisialisasi.")

@bot.message_handler(commands=['threshold'])
def change_threshold(message):
    global SPEED_THRESHOLD
    try:
        new_threshold = float(message.text.split()[1])
        if new_threshold > 0:
            old_threshold = SPEED_THRESHOLD
            SPEED_THRESHOLD = new_threshold
            response = f"Batas kecepatan diubah dari {old_threshold} menjadi {SPEED_THRESHOLD} km/jam."
            bot.reply_to(message, response)
            log_activity(response)
        else:
            bot.reply_to(message, "Batas kecepatan harus lebih dari 0.")
    except (IndexError, ValueError):
        bot.reply_to(message, "Format yang benar: /threshold [nilai dalam km/jam]")

@bot.message_handler(commands=['report'])
def request_report(message):
    send_activity_report()

@bot.message_handler(commands=['log'])
def send_log_file(message):
    try:
        with open("speed_detection_activity.log", 'rb') as log_file:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            bot.send_document(
                message.chat.id, 
                log_file, 
                caption=f"Log file as of {current_time}"
            )
    except Exception as e:
        bot.reply_to(message, f"Error sending log file: {str(e)}")

@bot.message_handler(commands=['help'])
def send_help(message):
    help_text = (
        "🚦 *SISTEM PEMANTAUAN KECEPATAN KENDARAAN* 🚦\n\n"
        "Perintah yang tersedia:\n"
        "/start - Memulai bot dan mendapatkan Chat ID\n"
        "/status - Melihat status sistem saat ini\n"
        "/report - Meminta laporan aktivitas terbaru\n"
        "/threshold [nilai] - Mengubah batas kecepatan (km/jam)\n"
        "/log - Mengirim file log aktivitas sistem\n"
        "/shutdown - Mematikan sistem dari jarak jauh\n"
        "/help - Menampilkan pesan bantuan ini"
    )
    bot.send_message(message.chat.id, help_text, parse_mode="Markdown")

# Jalankan bot dalam thread terpisah
def run_bot():
    try:
        log_activity("Telegram bot started")
        bot.polling(none_stop=True)
    except Exception as e:
        log_activity(f"Bot error: {str(e)}")

# Check camera status and send alerts if needed
def monitor_system_health():
    global system_running
    while system_running:
        current_time = time.time()
        try:
            for cam_name, stats in system_stats["cameras"].items():
                if stats["active"]:
                    if stats["last_frame_time"] and current_time - stats["last_frame_time"] > 5:
                        stats["active"] = False
                        log_activity(f"⚠️ {cam_name} tidak merespons selama 5 detik", send_to_telegram=True)
            
            time.sleep(5)
        except Exception as e:
            log_activity(f"Health monitor error: {str(e)}")
            time.sleep(10)

# Flag to signal system shutdown
system_running = True

@bot.message_handler(commands=['shutdown'])
def shutdown_system(message):
    global system_running
    
    bot.reply_to(message, "⚠️ Memulai proses shutdown sistem...")
    log_activity("Shutdown command received from Telegram", send_to_telegram=True)
    
    system_running = False
    
    bot.send_message(message.chat.id, "🔴 Sistem akan dimatikan dalam beberapa detik...")
    
    threading.Thread(target=lambda: (time.sleep(3), os._exit(0))).start()

# Modified process_camera with stationary vehicle detection
def process_camera(camera_index, window_name):
    global system_running
    log_activity(f"Starting camera {camera_index} as '{window_name}'", send_to_telegram=True)
    
    if window_name not in system_stats["cameras"]:
        system_stats["cameras"][window_name] = {
            "active": False,
            "vehicles_detected": 0,
            "violations": 0,
            "last_frame_time": None
        }
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        error_msg = f"Error: Kamera {camera_index} tidak bisa dibuka."
        log_activity(error_msg, send_to_telegram=True)
        system_stats["cameras"][window_name]["active"] = False
        return

    system_stats["cameras"][window_name]["active"] = True
    prev_positions = {}
    prev_timestamps = {}
    movement_counters = {}  # Track consecutive frames with movement
    last_frame_count = 0
    frame_count = 0
    fps_start_time = time.time()
    processing_times = []
    
    while cap.isOpened() and system_running:
        current_time = time.time()
        ret, frame = cap.read()
        if not ret:
            system_stats["cameras"][window_name]["active"] = False
            log_activity(f"Camera {window_name} disconnected", send_to_telegram=True)
            break

        frame_count += 1
        system_stats["cameras"][window_name]["last_frame_time"] = current_time
        
        if frame_count % 2 != 0:
            cv2.imshow(window_name, frame)
            continue

        # Measure inference time
        inference_start = time.time()
        with torch.no_grad():  # Disable gradient calculation
            results = model(frame)
        inference_time = time.time() - inference_start
        
        new_positions = {}
        vehicles_in_frame = 0

        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                class_id = int(cls)
                if conf > CONFIDENCE_THRESHOLD and class_id in COCO_CLASSES:
                    x1, y1, x2, y2 = map(int, box)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    label = COCO_CLASSES[class_id]

                    data = {
                        "bbox": (x1, y1, x2, y2),
                        "label": label,
                        "center": center
                    }

                    new_positions[len(new_positions)] = data
                    
                    if label in ["Mobil", "Motor"]:
                        vehicles_in_frame += 1

                    if label in ["Mobil", "Motor"]:
                        plate_start = time.time()
                        plate_number = detect_license_plate(frame, (x1, y1, x2, y2))
                        processing_times.append(time.time() - plate_start)
                        data["plate_number"] = plate_number
                        cv2.putText(frame, f"{label} - {plate_number}", (x1, y1 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if vehicles_in_frame > 0 and vehicles_in_frame != last_frame_count:
            system_stats["total_vehicles_detected"] += (vehicles_in_frame - last_frame_count)
            system_stats["cameras"][window_name]["vehicles_detected"] += (vehicles_in_frame - last_frame_count)
            last_frame_count = vehicles_in_frame

        matched_objects = match_objects(prev_positions, new_positions)

        for object_id, data in matched_objects.items():
            if isinstance(data, dict) and "bbox" in data and "center" in data:
                bbox = data["bbox"]
                label = data["label"]
                center = data["center"]
                x1, y1, x2, y2 = bbox

                if object_id in prev_positions:
                    prev_center = prev_positions[object_id]["center"]
                    prev_time = prev_timestamps.get(object_id, current_time)
                    time_delta = current_time - prev_time

                    if label in ["Mobil", "Motor"]:
                        # Calculate displacement
                        displacement = np.sqrt((center[0] - prev_center[0])**2 + 
                                             (center[1] - prev_center[1])**2)
                        
                        # Initialize movement counter if needed
                        if object_id not in movement_counters:
                            movement_counters[object_id] = 0
                        
                        # Check if vehicle is moving significantly
                        if displacement > MIN_DISPLACEMENT_PX:
                            movement_counters[object_id] += 1
                        else:
                            movement_counters[object_id] = max(0, movement_counters[object_id] - 1)
                        
                        # Only calculate speed if vehicle has been moving for consecutive frames
                        if movement_counters[object_id] >= MIN_MOVEMENT_FRAMES:
                            speed = calculate_speed(prev_center, center, time_delta, DISTANCE_PER_PIXEL)
                        else:
                            speed = 0.0
                        
                        plate_number = data.get("plate_number", "Unknown")
                        
                        color = (0, 0, 255) if speed > SPEED_THRESHOLD else (255, 255, 255)
                        cv2.putText(frame, f"{speed:.2f} km/h", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Send notification only if vehicle is moving and exceeds threshold
                        if speed > SPEED_THRESHOLD:
                            if (plate_number not in last_notification or 
                                current_time - last_notification.get(plate_number, 0) > NOTIFICATION_COOLDOWN):
                                notification_frame = frame.copy()
                                cv2.rectangle(notification_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(notification_frame, f"SPEEDING: {speed:.2f} km/h", 
                                          (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                
                                threading.Thread(
                                    target=send_telegram_notification,
                                    args=(window_name, label, plate_number, speed, notification_frame)
                                ).start()
                                
                                last_notification[plate_number] = current_time
                                system_stats["cameras"][window_name]["violations"] += 1

                prev_timestamps[object_id] = current_time

        prev_positions = new_positions
        cv2.imshow(window_name, frame)
        
        # Log performance metrics
        if frame_count % 50 == 0:
            elapsed = time.time() - fps_start_time
            actual_fps = 50 / elapsed
            log_activity(f"Camera {window_name} FPS: {actual_fps:.1f}, Inference: {inference_time:.3f}s")
            fps_start_time = time.time()
            frame_count = 0
            
            if processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                log_activity(f"Avg plate processing: {avg_time:.3f}s")
                processing_times = []
        
        if current_time - system_stats["last_activity_report"] > ACTIVITY_LOG_INTERVAL:
            threading.Thread(target=send_activity_report).start()
        
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    system_stats["cameras"][window_name]["active"] = False
    log_activity(f"Camera {window_name} stopped", send_to_telegram=True)
    cap.release()
    cv2.destroyWindow(window_name)

# Jalankan semua komponen
if __name__ == "__main__":
    system_stats["start_time"] = datetime.now()
    system_stats["last_activity_report"] = time.time()
    
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()
    
    health_monitor = threading.Thread(target=monitor_system_health)
    health_monitor.daemon = True
    health_monitor.start()
    
    startup_message = "✅ Sistem Pemantauan Kecepatan Kendaraan dimulai."
    log_activity(startup_message, send_to_telegram=True)
    
    try:
        log_activity("Memulai pemantauan kamera...")
        thread1 = threading.Thread(target=process_camera, args=(0, "Kamera 1"))
        thread2 = threading.Thread(target=process_camera, args=(1, "Kamera 2"))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()
    except KeyboardInterrupt:
        system_running = False
        shutdown_message = "Program dihentikan oleh pengguna."
        log_activity(shutdown_message, send_to_telegram=True)
    except Exception as e:
        system_running = False
        error_message = f"Kesalahan sistem: {str(e)}"
        log_activity(error_message, send_to_telegram=True)
    finally:
        send_activity_report()
        log_activity("Sistem dimatikan.", send_to_telegram=True)

        cv2.destroyAllWindows()
        bot.stop_polling()
        log_activity("Bot polling stopped.")
        os._exit(0)