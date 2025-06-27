import cv2
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import threading
import telebot
import time
import os
import logging
import firebase_admin
from firebase_admin import credentials, db
import uuid
import torch
import queue
import socket
import requests
import csv
import glob

# Configuration flags
TEST_MODE = False  # Set to True for testing with video files
DATASET_PATH = r"C:\Users\thega\OneDrive\Documents\Speed Detection Program\Speed detection YOLO\dataset\Video"
DISPLAY_RESULTS = True  # Set to True to show detection windows

# Inisialisasi Firebase
try:
    cred = credentials.Certificate("speed-detection-109da-firebase-adminsdk-fbsvc-d9120017d2.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://speed-detection-109da-default-rtdb.asia-southeast1.firebasedatabase.app/',
    })
    firebase_initialized = True
except Exception as e:
    print(f"Firebase initialization failed: {e}")
    firebase_initialized = False

# Initialize Firebase references if available
if firebase_initialized:
    violations_ref = db.reference('violations')
    system_stats_ref = db.reference('system_stats')
    cameras_ref = db.reference('cameras')
    daily_stats_ref = db.reference('daily_stats')
    people_ref = db.reference('people_detections')
else:
    # Create dummy references to prevent crashes
    class DummyRef:
        def child(self, *args, **kwargs): return self
        def set(self, *args, **kwargs): pass
        def transaction(self, *args, **kwargs): pass
    violations_ref = system_stats_ref = cameras_ref = daily_stats_ref = people_ref = DummyRef()

# Telegram Bot Setup
BOT_TOKEN = "7633151627:AAFowoEJTa9In8nYpHccAi9fSBP92Vw5lik"
CHAT_ID = "6451128792"
bot = None
telegram_available = not TEST_MODE  # Disable Telegram in test mode

if not TEST_MODE:
    try:
        bot = telebot.TeleBot(BOT_TOKEN)
        telegram_available = True
    except Exception as e:
        print(f"Telegram bot initialization failed: {e}")
        telegram_available = False

# Konstanta
FPS = 25
DISTANCE_PER_PIXEL = 0.0057
SPEED_THRESHOLD = 30
CONFIDENCE_THRESHOLD = 0.5
NOTIFICATION_COOLDOWN = 10
ACTIVITY_LOG_INTERVAL = 60
MIN_DISPLACEMENT_PX = 5
MIN_MOVEMENT_FRAMES = 3
PERSON_COOLDOWN = 30
MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds between retries

# Network status tracking
network_available = True
last_network_check = 0
NETWORK_CHECK_INTERVAL = 30  # seconds

# Set up logging
logger = logging.getLogger("SpeedDetectionSystem")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler("speed_detection_activity.log", encoding='utf-8')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Inisialisasi model
model = YOLO("yolov8n.pt").to('cpu')
torch.set_num_threads(4)

# Kelas objek yang ingin dideteksi
COCO_CLASSES = {
    0: "Orang",
    2: "Mobil",
    3: "Motor"
}

# Tracking last notification time
last_notification = {}
last_person_notification = {}

# Global statistics dictionary
system_stats = {
    "start_time": None,
    "cameras": {},
    "total_vehicles_detected": 0,
    "total_violations": 0,
    "total_people_detected": 0,
    "last_activity_report": 0,
    "pending_notifications": 0,
    "pending_person_notifications": 0
}

# Notification queues
violation_queue = queue.Queue()
person_queue = queue.Queue()

# ================== CSV LOGGING SETUP ==================
CSV_LOG_FILE = "detection_log.csv"
ACCURACY_LOG_FILE = "speed_accuracy.csv"
csv_lock = threading.Lock()

def init_csv_log():
    """Initialize CSV log file with headers if it doesn't exist"""
    if not os.path.exists(CSV_LOG_FILE):
        with open(CSV_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", 
                "Camera", 
                "ObjectType", 
                "Speed(km/h)", 
                "ViolationStatus", 
                "ViolationDescription",
                "Confidence",
                "BoundingBox"
            ])
    
    # Initialize accuracy log
    if not os.path.exists(ACCURACY_LOG_FILE):
        with open(ACCURACY_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "VideoFile",
                "ExpectedSpeed(km/h)",
                "DetectedSpeed(km/h)",
                "DetectionTime",
                "ObjectType",
                "Confidence",
                "ErrorPercentage"
            ])

def log_to_csv(camera_name, object_type, speed, violation_status, violation_description, confidence, bbox):
    """Log detection event to CSV file"""
    timestamp = datetime.now().isoformat()
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}" if bbox else ""
    
    # Format violation description
    if not violation_description and violation_status:
        violation_description = "Exceeded speed limit" if object_type != "Orang" else "Orang terdeteksi saat malam hari"
    
    row = [
        timestamp,
        camera_name,
        object_type,
        round(speed, 2) if speed is not None else "",
        "True" if violation_status else "False",
        violation_description,
        round(confidence, 4),
        bbox_str
    ]
    
    with csv_lock:
        with open(CSV_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

def log_accuracy(video_file, expected_speed, detected_speed, detection_time, object_type, confidence):
    """Log accuracy data to CSV file"""
    try:
        expected_speed = float(expected_speed)
        detected_speed = float(detected_speed)
        error_percentage = abs((detected_speed - expected_speed) / expected_speed) * 100
        
        row = [
            os.path.basename(video_file),
            expected_speed,
            detected_speed,
            detection_time,
            object_type,
            confidence,
            error_percentage
        ]
        
        with csv_lock:
            with open(ACCURACY_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
    except Exception as e:
        log_activity(f"Error logging accuracy data: {str(e)}")
# ================== END OF CSV LOGGING ==================

def check_network_connection():
    """Check if we have an active network connection"""
    global network_available
    try:
        # Try to connect to a reliable service
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        network_available = True
        return True
    except OSError:
        try:
            # Try an alternative check
            requests.get("http://www.google.com", timeout=5)
            network_available = True
            return True
        except:
            network_available = False
            return False

def save_violation_to_firebase(camera_name, vehicle_type, speed):
    """Save violation metadata to Firebase with retry logic"""
    if not firebase_initialized or TEST_MODE:
        return None
        
    for attempt in range(MAX_RETRIES):
        try:
            violation_id = str(uuid.uuid4())
            current_time = datetime.now()
            
            violation_data = {
                'id': violation_id,
                'camera_name': camera_name,
                'vehicle_type': vehicle_type,
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
            log_activity(f"Error saving violation to Firebase (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
            time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
            
    log_activity("Failed to save violation to Firebase after multiple attempts")
    return None

def save_person_detection_to_firebase(camera_name):
    """Save person detection metadata to Firebase with retry logic"""
    if not firebase_initialized or TEST_MODE:
        return None
        
    for attempt in range(MAX_RETRIES):
        try:
            detection_id = str(uuid.uuid4())
            current_time = datetime.now()
            
            detection_data = {
                'id': detection_id,
                'camera_name': camera_name,
                'capture_time': current_time.isoformat(),
                'timestamp': int(time.time()),
                'location': {
                    'camera_id': camera_name.lower().replace(' ', '_'),
                    'coordinates': None
                },
                'notified': True,
                'created_at': current_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            people_ref.child(detection_id).set(detection_data)
            
            log_activity(f"Person detection saved to Firebase with ID: {detection_id}")
            return detection_id
        except Exception as e:
            log_activity(f"Error saving person detection to Firebase (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
            time.sleep(RETRY_DELAY * (2 ** attempt))
            
    log_activity("Failed to save person detection to Firebase after multiple attempts")
    return None

def update_system_stats_firebase():
    """Update system statistics in Firebase with retry logic"""
    if not firebase_initialized or TEST_MODE:
        return
        
    for attempt in range(MAX_RETRIES):
        try:
            current_time = datetime.now()
            uptime = current_time - system_stats["start_time"] if system_stats["start_time"] else None
            
            firebase_stats = {
                'start_time': system_stats["start_time"].isoformat() if system_stats["start_time"] else None,
                'uptime_seconds': int(uptime.total_seconds()) if uptime else 0,
                'total_vehicles_detected': system_stats["total_vehicles_detected"],
                'total_violations': system_stats["total_violations"],
                'total_people_detected': system_stats["total_people_detected"],
                'last_update': current_time.isoformat(),
                'speed_threshold': SPEED_THRESHOLD,
                'system_status': 'active',
                'pending_notifications': system_stats["pending_notifications"],
                'pending_person_notifications': system_stats["pending_person_notifications"]
            }
            
            system_stats_ref.set(firebase_stats)
            
            # Update camera statuses
            for cam_name, stats in system_stats["cameras"].items():
                camera_data = {
                    'name': cam_name,
                    'active': stats["active"],
                    'vehicles_detected': stats["vehicles_detected"],
                    'violations': stats["violations"],
                    'people_detected': stats.get("people_detected", 0),
                    'last_frame_time': stats["last_frame_time"],
                    'last_update': current_time.isoformat()
                }
                cameras_ref.child(cam_name.lower().replace(' ', '_')).set(camera_data)
                
            return
        except Exception as e:
            log_activity(f"Error updating Firebase stats (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
            time.sleep(RETRY_DELAY * (2 ** attempt))
            
    log_activity("Failed to update Firebase stats after multiple attempts")

def log_activity(message, send_to_telegram=False):
    """Log activity to file and optionally send to Telegram"""
    logger.info(message)
    if send_to_telegram and telegram_available and CHAT_ID and not TEST_MODE:
        try:
            bot.send_message(chat_id=CHAT_ID, text=message)
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {str(e)}")

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

def is_night_time():
    """Check if current time is during night hours (6 PM to 6 AM)"""
    current_hour = datetime.now().hour
    return current_hour >= 18 or current_hour < 6

def send_telegram_notification(camera_name, vehicle_type, speed, frame):
    """Send notification with retry and queuing capabilities"""
    if TEST_MODE:
        return  # Skip notifications during testing
        
    # Save the frame to disk
    violation_dir = "violation_images"
    if not os.path.exists(violation_dir):
        os.makedirs(violation_dir)
    
    image_path = f"{violation_dir}/violation_{int(time.time())}.jpg"
    cv2.imwrite(image_path, frame)
    
    # Prepare notification data
    notification_data = {
        "camera_name": camera_name,
        "vehicle_type": vehicle_type,
        "speed": speed,
        "image_path": image_path,
        "timestamp": time.time(),
        "attempts": 0
    }
    
    # Add to queue
    violation_queue.put(notification_data)
    system_stats["pending_notifications"] += 1
    log_activity(f"Violation queued for {vehicle_type} at {speed:.2f} km/h on {camera_name}")

def send_person_notification(camera_name, frame):
    """Send person notification with retry and queuing capabilities"""
    if TEST_MODE:
        return  # Skip notifications during testing
        
    # Save the frame to disk
    person_dir = "person_images"
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    
    image_path = f"{person_dir}/person_{int(time.time())}.jpg"
    cv2.imwrite(image_path, frame)
    
    # Prepare notification data
    notification_data = {
        "camera_name": camera_name,
        "image_path": image_path,
        "timestamp": time.time(),
        "attempts": 0
    }
    
    # Add to queue
    person_queue.put(notification_data)
    system_stats["pending_person_notifications"] += 1
    log_activity(f"Person detection queued for {camera_name}")

def process_notification_queues():
    """Process queued notifications when network is available"""
    global network_available
    
    while True:
        try:
            # Check network status periodically
            current_time = time.time()
            if current_time - last_network_check > NETWORK_CHECK_INTERVAL:
                network_available = check_network_connection()
                last_network_check = current_time
                
                if network_available:
                    log_activity("Network connection restored")
                else:
                    log_activity("Network connection unavailable")
            
            # Process violation queue
            if not violation_queue.empty() and network_available and not TEST_MODE:
                notification_data = violation_queue.get()
                
                # Prepare message
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                message = f"⚠️ PERINGATAN KECEPATAN BERLEBIH! ⚠️\n\n"
                message += f"📹 Kamera: {notification_data['camera_name']}\n"
                message += f"🚗 Tipe Kendaraan: {notification_data['vehicle_type']}\n"
                message += f"⚡ Kecepatan: {notification_data['speed']:.2f} km/jam\n"
                message += f"⏰ Waktu: {current_time}\n"
                message += f"\nKendaraan melebihi batas kecepatan {SPEED_THRESHOLD} km/jam!"
                
                # Try to send
                try:
                    with open(notification_data["image_path"], 'rb') as photo:
                        bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=message)
                    
                    # Save to Firebase
                    violation_id = save_violation_to_firebase(
                        notification_data["camera_name"],
                        notification_data["vehicle_type"],
                        notification_data["speed"]
                    )
                    
                    # Update message with ID if available
                    if violation_id:
                        update_message = message + f"\n🆔 ID Pelanggaran: {violation_id}"
                        try:
                            bot.edit_message_caption(
                                chat_id=CHAT_ID,
                                message_id=bot.last_message_id,
                                caption=update_message
                            )
                        except:
                            pass
                    
                    log_activity(f"Violation notification sent for {notification_data['vehicle_type']}")
                    system_stats["pending_notifications"] -= 1
                    system_stats["total_violations"] += 1
                except Exception as e:
                    # Retry later
                    notification_data["attempts"] += 1
                    if notification_data["attempts"] < MAX_RETRIES:
                        violation_queue.put(notification_data)
                        log_activity(f"Failed to send violation notification (retry {notification_data['attempts']}): {str(e)}")
                    else:
                        log_activity(f"Permanently failed to send violation notification: {str(e)}")
                        system_stats["pending_notifications"] -= 1
                        try:
                            os.remove(notification_data["image_path"])
                        except:
                            pass
            
            # Process person queue
            if not person_queue.empty() and network_available and not TEST_MODE:
                notification_data = person_queue.get()
                
                # Prepare message
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                message = f"⚠️ ORANG TERDETEKSI PADA MALAM HARI! ⚠️\n\n"
                message += f"📹 Kamera: {notification_data['camera_name']}\n"
                message += f"⏰ Waktu: {current_time}\n"
                message += f"\nOrang terdeteksi di area pada malam hari."
                
                # Try to send
                try:
                    with open(notification_data["image_path"], 'rb') as photo:
                        bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=message)
                    
                    # Save to Firebase
                    detection_id = save_person_detection_to_firebase(
                        notification_data["camera_name"]
                    )
                    
                    # Update message with ID if available
                    if detection_id:
                        update_message = message + f"\n🆔 ID Deteksi: {detection_id}"
                        try:
                            bot.edit_message_caption(
                                chat_id=CHAT_ID,
                                message_id=bot.last_message_id,
                                caption=update_message
                            )
                        except:
                            pass
                    
                    log_activity(f"Person notification sent for {notification_data['camera_name']}")
                    system_stats["pending_person_notifications"] -= 1
                    system_stats["total_people_detected"] += 1
                except Exception as e:
                    # Retry later
                    notification_data["attempts"] += 1
                    if notification_data["attempts"] < MAX_RETRIES:
                        person_queue.put(notification_data)
                        log_activity(f"Failed to send person notification (retry {notification_data['attempts']}): {str(e)}")
                    else:
                        log_activity(f"Permanently failed to send person notification: {str(e)}")
                        system_stats["pending_person_notifications"] -= 1
                        try:
                            os.remove(notification_data["image_path"])
                        except:
                            pass
            
            # Sleep briefly to avoid high CPU usage
            time.sleep(1)
            
        except Exception as e:
            log_activity(f"Error in notification queue processing: {str(e)}")
            time.sleep(5)

def send_activity_report():
    """Send a comprehensive activity report to Telegram"""
    if not telegram_available or not CHAT_ID or TEST_MODE:
        return
    
    try:
        current_time = datetime.now()
        uptime = current_time - system_stats["start_time"]
        uptime_str = str(uptime).split('.')[0]
        
        # Create activity report
        message = f"📊 SISTEM PEMANTAUAN KECEPATAN - LAPORAN AKTIVITAS 📊\n\n"
        message += f"⏱️ Waktu Operasi: {uptime_str}\n"
        message += f"🚗 Total Kendaraan Terdeteksi: {system_stats['total_vehicles_detected']}\n"
        message += f"🚨 Total Pelanggaran: {system_stats['total_violations']}\n"
        message += f"👤 Total Orang Terdeteksi (Malam): {system_stats['total_people_detected']}\n"
        message += f"⏳ Pending Notifications: {system_stats['pending_notifications']}\n"
        message += f"⏳ Pending Person Notifications: {system_stats['pending_person_notifications']}\n\n"
        
        # Add camera-specific stats
        message += "📷 STATUS KAMERA:\n"
        for cam_name, stats in system_stats["cameras"].items():
            status = "✅ Aktif" if stats["active"] else "❌ Tidak Aktif"
            message += f"- {cam_name}: {status}\n"
            message += f"  Kendaraan: {stats['vehicles_detected']}\n"
            message += f"  Pelanggaran: {stats['violations']}\n"
            message += f"  Orang (Malam): {stats.get('people_detected', 0)}\n"
        
        message += f"\n⚙️ Batas Kecepatan: {SPEED_THRESHOLD} km/jam"
        message += f"\n🌐 Network Status: {'✅ Available' if network_available else '❌ Unavailable'}"
        
        bot.send_message(chat_id=CHAT_ID, text=message)
        system_stats["last_activity_report"] = time.time()
        log_activity("Activity report sent to Telegram")
    except Exception as e:
        log_activity(f"Failed to send activity report: {str(e)}")

# Telegram bot commands
def handle_telegram_commands():
    """Process Telegram commands with resilience to network issues"""
    global telegram_available
    
    while True:
        try:
            if not telegram_available or TEST_MODE:
                time.sleep(30)
                continue
            
            if not telegram_available:
                # Try to reinitialize Telegram bot
                try:
                    global bot
                    bot = telebot.TeleBot(BOT_TOKEN)
                    telegram_available = True
                    log_activity("Telegram bot reinitialized successfully")
                except Exception as e:
                    log_activity(f"Failed to reinitialize Telegram bot: {str(e)}")
                    time.sleep(30)
                    continue
            
            @bot.message_handler(commands=['start'])
            def send_welcome(message):
                try:
                    bot.reply_to(message, "Selamat datang di Sistem Pemantauan Kecepatan Kendaraan! Bot siap memantau.")
                    global CHAT_ID
                    CHAT_ID = message.chat.id
                    bot.send_message(chat_id=CHAT_ID, text=f"Chat ID Anda adalah: {CHAT_ID}")
                    log_activity(f"Bot started by user with chat ID: {CHAT_ID}")
                except Exception as e:
                    log_activity(f"Failed to process /start command: {str(e)}")

            @bot.message_handler(commands=['status'])
            def send_status(message):
                try:
                    if system_stats["start_time"]:
                        send_activity_report()
                    else:
                        bot.reply_to(message, "Sistem belum diinisialisasi.")
                except Exception as e:
                    log_activity(f"Failed to process /status command: {str(e)}")

            @bot.message_handler(commands=['threshold'])
            def change_threshold(message):
                try:
                    global SPEED_THRESHOLD
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
                except Exception as e:
                    log_activity(f"Failed to process /threshold command: {str(e)}")

            @bot.message_handler(commands=['report'])
            def request_report(message):
                try:
                    send_activity_report()
                except Exception as e:
                    log_activity(f"Failed to process /report command: {str(e)}")

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
                    try:
                        bot.reply_to(message, f"Error sending log file: {str(e)}")
                    except:
                        log_activity(f"Failed to send log file: {str(e)}")

            @bot.message_handler(commands=['csvlog'])
            def send_csv_log(message):
                try:
                    with open(CSV_LOG_FILE, 'rb') as csv_file:
                        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                        bot.send_document(
                            message.chat.id, 
                            csv_file, 
                            caption=f"Detection Log CSV as of {current_time}"
                        )
                    log_activity("CSV log file sent to Telegram")
                except Exception as e:
                    try:
                        bot.reply_to(message, f"Error sending CSV log: {str(e)}")
                    except:
                        log_activity(f"Failed to send CSV log: {str(e)}")
                        
            @bot.message_handler(commands=['accuracy'])
            def send_accuracy_log(message):
                try:
                    with open(ACCURACY_LOG_FILE, 'rb') as csv_file:
                        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                        bot.send_document(
                            message.chat.id, 
                            csv_file, 
                            caption=f"Accuracy Log CSV as of {current_time}"
                        )
                    log_activity("Accuracy log file sent to Telegram")
                except Exception as e:
                    try:
                        bot.reply_to(message, f"Error sending accuracy log: {str(e)}")
                    except:
                        log_activity(f"Failed to send accuracy log: {str(e)}")

            @bot.message_handler(commands=['help'])
            def send_help(message):
                try:
                    help_text = (
                        "🚦 *SISTEM PEMANTAUAN KECEPATAN KENDARAAN* 🚦\n\n"
                        "Perintah yang tersedia:\n"
                        "/start - Memulai bot dan mendapatkan Chat ID\n"
                        "/status - Melihat status sistem saat ini\n"
                        "/report - Meminta laporan aktivitas terbaru\n"
                        "/threshold [nilai] - Mengubah batas kecepatan (km/jam)\n"
                        "/log - Mengirim file log aktivitas sistem\n"
                        "/csvlog - Mengirim file log deteksi dalam format CSV\n"
                        "/accuracy - Mengirim file log akurasi kecepatan\n"
                        "/shutdown - Mematikan sistem dari jarak jauh\n"
                        "/help - Menampilkan pesan bantuan ini"
                    )
                    bot.send_message(message.chat.id, help_text, parse_mode="Markdown")
                except Exception as e:
                    log_activity(f"Failed to process /help command: {str(e)}")

            @bot.message_handler(commands=['shutdown'])
            def shutdown_system(message):
                try:
                    global system_running
                    bot.reply_to(message, "⚠️ Memulai proses shutdown sistem...")
                    log_activity("Shutdown command received from Telegram", send_to_telegram=True)
                    system_running = False
                    bot.send_message(message.chat.id, "🔴 Sistem akan dimatikan dalam beberapa detik...")
                    threading.Thread(target=lambda: (time.sleep(3), os._exit(0))).start()
                except Exception as e:
                    log_activity(f"Failed to process /shutdown command: {str(e)}")
            
            # Start polling
            log_activity("Telegram bot started")
            bot.polling(none_stop=True)
            
        except Exception as e:
            log_activity(f"Telegram bot error: {str(e)}")
            telegram_available = False
            time.sleep(30)

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

# Process video source (camera index or video file)
def process_video_source(video_source, camera_name):
    global system_running
    
    # Determine if we're processing a camera or video file
    if isinstance(video_source, int):
        log_activity(f"Starting camera {video_source} as '{camera_name}'", send_to_telegram=not TEST_MODE)
        expected_speed = None
    else:
        log_activity(f"Processing video: {video_source} as '{camera_name}'")
        # Extract expected speed from folder name
        folder_name = os.path.basename(os.path.dirname(video_source))
        try:
            # Extract numbers from folder name (e.g., "10km" -> 10)
            expected_speed = int(''.join(filter(str.isdigit, folder_name)))
        except:
            expected_speed = None
            log_activity(f"Could not extract expected speed from folder name: {folder_name}")
    
    if DISPLAY_RESULTS:
        cv2.namedWindow(camera_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(camera_name, 800, 600)
    
    if camera_name not in system_stats["cameras"]:
        system_stats["cameras"][camera_name] = {
            "active": False,
            "vehicles_detected": 0,
            "violations": 0,
            "people_detected": 0,
            "last_frame_time": None
        }
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        error_msg = f"Error: Video source {video_source} tidak bisa dibuka."
        log_activity(error_msg)
        system_stats["cameras"][camera_name]["active"] = False
        if DISPLAY_RESULTS:
            cv2.destroyWindow(camera_name)
        return

    system_stats["cameras"][camera_name]["active"] = True
    prev_positions = {}
    prev_timestamps = {}
    movement_counters = {}  # Track consecutive frames with movement
    last_frame_count = 0
    frame_count = 0
    fps_start_time = time.time()
    
    while cap.isOpened() and system_running:
        current_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break  # End of video or camera disconnected

        frame_count += 1
        system_stats["cameras"][camera_name]["last_frame_time"] = current_time
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Measure inference time
        inference_start = time.time()
        with torch.no_grad():  # Disable gradient calculation
            results = model(frame)
        inference_time = time.time() - inference_start
        
        new_positions = {}
        vehicles_in_frame = 0
        person_detected = False  # Track if person is detected

        # Process detection results
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
                        "center": center,
                        "confidence": float(conf)
                    }

                    new_positions[len(new_positions)] = data
                    
                    # Draw bounding box and label
                    color = (0, 255, 0)  # Default green
                    if label == "Orang":
                        color = (0, 255, 255)  # Yellow for people
                        night = is_night_time()
                        log_to_csv(
                            camera_name=camera_name,
                            object_type=label,
                            speed=None,
                            violation_status=night,
                            violation_description="Orang terdeteksi saat malam hari" if night else "",
                            confidence=float(conf),
                            bbox=(x1, y1, x2, y2))
                        if night:
                            person_detected = True
                            system_stats["cameras"][camera_name]["people_detected"] += 1
                    elif label in ["Mobil", "Motor"]:
                        color = (0, 0, 255)  # Red for vehicles
                    
                    if DISPLAY_RESULTS:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, f"{label} {conf:.2f}", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if label in ["Mobil", "Motor"]:
                        vehicles_in_frame += 1

        # Handle person detection at night
        if person_detected and is_night_time():
            # Create a simple camera ID for cooldown
            camera_id = f"{camera_name}"
            
            # Check cooldown for this camera
            if (camera_id not in last_person_notification or 
                current_time - last_person_notification.get(camera_id, 0) > PERSON_COOLDOWN):
                
                notification_frame = frame.copy()
                
                # Highlight all detected people
                for r in results:
                    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                        class_id = int(cls)
                        if class_id == 0 and conf > CONFIDENCE_THRESHOLD:  # Person
                            x1, y1, x2, y2 = map(int, box)
                            if DISPLAY_RESULTS:
                                cv2.rectangle(notification_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                                cv2.putText(notification_frame, "ORANG", 
                                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Queue the notification
                threading.Thread(
                    target=send_person_notification,
                    args=(camera_name, notification_frame)
                ).start()
                
                last_person_notification[camera_id] = current_time

        if vehicles_in_frame > 0 and vehicles_in_frame != last_frame_count:
            system_stats["total_vehicles_detected"] += (vehicles_in_frame - last_frame_count)
            system_stats["cameras"][camera_name]["vehicles_detected"] += (vehicles_in_frame - last_frame_count)
            last_frame_count = vehicles_in_frame

        matched_objects = match_objects(prev_positions, new_positions)

        # Process tracked objects
        for object_id, data in matched_objects.items():
            if isinstance(data, dict) and "bbox" in data and "center" in data:
                bbox = data["bbox"]
                label = data["label"]
                center = data["center"]
                confidence = data.get("confidence", 0.0)
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
                        
                        # Log accuracy data (only for video files in test mode)
                        if TEST_MODE and expected_speed is not None and speed > 0:
                            log_accuracy(
                                video_source,
                                expected_speed,
                                speed,
                                datetime.now().isoformat(),
                                label,
                                confidence
                            )
                        
                        # Log to detection CSV
                        violation = speed > SPEED_THRESHOLD
                        log_to_csv(
                            camera_name=camera_name,
                            object_type=label,
                            speed=speed,
                            violation_status=violation,
                            violation_description="",
                            confidence=confidence,
                            bbox=bbox
                        )
                        
                        # Display speed on the frame
                        if DISPLAY_RESULTS:
                            speed_text = f"{speed:.1f} km/h"
                            speed_color = (0, 255, 0)  # Green
                            if violation:
                                speed_color = (0, 0, 255)  # Red for speeding
                            cv2.putText(display_frame, speed_text, 
                                      (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, speed_color, 2)
                        
                        # Send notification only if vehicle is moving and exceeds threshold
                        if violation and not TEST_MODE:
                            # Create a simple vehicle ID for cooldown
                            vehicle_id = f"{camera_name}_{object_id}"
                            
                            if (vehicle_id not in last_notification or 
                                current_time - last_notification.get(vehicle_id, 0) > NOTIFICATION_COOLDOWN):
                                notification_frame = frame.copy()
                                cv2.rectangle(notification_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(notification_frame, f"SPEEDING: {speed:.2f} km/h", 
                                          (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                
                                # Queue the notification
                                threading.Thread(
                                    target=send_telegram_notification,
                                    args=(camera_name, label, speed, notification_frame)
                                ).start()
                                
                                last_notification[vehicle_id] = current_time
                                system_stats["cameras"][camera_name]["violations"] += 1

                prev_timestamps[object_id] = current_time

        prev_positions = new_positions
        
        # Display the processed frame
        if DISPLAY_RESULTS:
            cv2.imshow(camera_name, display_frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('x'):
                break
            elif key & 0xFF == ord('s'):
                send_activity_report()
        
        # Log performance metrics
        if frame_count % 50 == 0:
            elapsed = time.time() - fps_start_time
            actual_fps = 50 / elapsed
            log_activity(f"Camera {camera_name} FPS: {actual_fps:.1f}, Inference: {inference_time:.3f}s")
            fps_start_time = time.time()
            frame_count = 0
        
        if current_time - system_stats["last_activity_report"] > ACTIVITY_LOG_INTERVAL:
            threading.Thread(target=send_activity_report).start()

    system_stats["cameras"][camera_name]["active"] = False
    log_activity(f"Finished processing video source: {video_source}")
    cap.release()
    if DISPLAY_RESULTS:
        cv2.destroyWindow(camera_name)

def process_dataset():
    """Process all videos in the dataset folder structure"""
    # Walk through all folders in the dataset path
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            # Check for video files
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                # Use folder name as camera name for context
                folder_name = os.path.basename(root)
                camera_name = f"{folder_name}_{os.path.splitext(file)[0]}"
                process_video_source(video_path, camera_name)

# Jalankan semua komponen
if __name__ == "__main__":
    system_stats["start_time"] = datetime.now()
    system_stats["last_activity_report"] = time.time()
    
    # Initialize network status
    network_available = check_network_connection()
    last_network_check = time.time()
    
    # Initialize CSV logging
    init_csv_log()
    log_activity("CSV logging initialized")
    
    if not TEST_MODE:
        # Start Telegram command handler
        telegram_thread = threading.Thread(target=handle_telegram_commands)
        telegram_thread.daemon = True
        telegram_thread.start()
        
        # Start notification queue processor
        queue_processor = threading.Thread(target=process_notification_queues)
        queue_processor.daemon = True
        queue_processor.start()
        
        # Start system health monitor
        health_monitor = threading.Thread(target=monitor_system_health)
        health_monitor.daemon = True
        health_monitor.start()
    
    startup_message = "✅ Sistem Pemantauan Kecepatan Kendaraan dimulai."
    log_activity(startup_message, send_to_telegram=not TEST_MODE)
    
    # Periodically update Firebase stats
    def update_firebase_stats():
        while system_running and not TEST_MODE:
            try:
                update_system_stats_firebase()
                time.sleep(60)  # Update every minute
            except Exception as e:
                log_activity(f"Error updating Firebase stats: {str(e)}")
                time.sleep(30)
    
    if not TEST_MODE:
        firebase_updater = threading.Thread(target=update_firebase_stats)
        firebase_updater.daemon = True
        firebase_updater.start()
    
    try:
        if TEST_MODE:
            log_activity("Memulai pemrosesan dataset video...")
            process_dataset()
        else:
            log_activity("Memulai pemantauan kamera...")
            thread1 = threading.Thread(target=process_video_source, args=(0, "Kamera 1"))
            thread2 = threading.Thread(target=process_video_source, args=(1, "Kamera 2"))

            thread1.start()
            thread2.start()

            thread1.join()
            thread2.join()
    except KeyboardInterrupt:
        system_running = False
        shutdown_message = "Program dihentikan oleh pengguna."
        log_activity(shutdown_message, send_to_telegram=not TEST_MODE)
    except Exception as e:
        system_running = False
        error_message = f"Kesalahan sistem: {str(e)}"
        log_activity(error_message, send_to_telegram=not TEST_MODE)
    finally:
        if not TEST_MODE:
            send_activity_report()
        log_activity("Sistem dimatikan.", send_to_telegram=not TEST_MODE)

        try:
            if bot:
                bot.stop_polling()
        except:
            pass
        log_activity("Bot polling stopped.")
        if DISPLAY_RESULTS:
            cv2.destroyAllWindows()
        os._exit(0)