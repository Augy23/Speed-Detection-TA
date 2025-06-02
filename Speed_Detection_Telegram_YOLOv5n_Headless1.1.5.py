import cv2
import torch
import numpy as np
import easyocr
import threading
import telebot
import time
import os
import logging
from datetime import datetime
from PIL import Image

# Telegram Bot Setup
BOT_TOKEN = "7633151627:AAFowoEJTa9In8nYpHccAi9fSBP92Vw5lik"
CHAT_ID = "6451128792"
bot = telebot.TeleBot(BOT_TOKEN)

# Constants - Adjusted for 5.2m height and 60° tilt
FPS = 30
DISTANCE_PER_PIXEL = 0.008  # Adjusted for 5.2m height with 60° tilt
SPEED_THRESHOLD = 30
CONFIDENCE_THRESHOLD = 0.5
NOTIFICATION_COOLDOWN = 10
ACTIVITY_LOG_INTERVAL = 60

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("speed_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SpeedDetection")

# Load YOLOv5n model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.conf = CONFIDENCE_THRESHOLD

# OCR Reader
reader = easyocr.Reader(['en'], gpu=False)

# Vehicle classes from COCO
VEHICLE_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
TRACKED_VEHICLES = ['car', 'motorcycle', 'bus', 'truck']

# Global variables
last_notification = {}
system_stats = {
    "start_time": datetime.now(),
    "cameras": {},
    "total_vehicles": 0,
    "total_violations": 0,
    "last_activity_report": time.time()
}
system_running = True

def log_activity(message, send_telegram=False):
    """Log activity and optionally send to Telegram"""
    logger.info(message)
    if send_telegram and CHAT_ID:
        try:
            bot.send_message(chat_id=CHAT_ID, text=message)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

def calculate_speed(prev_center, curr_center, fps, distance_per_pixel):
    """Calculate vehicle speed"""
    distance_px = np.sqrt((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2)
    distance_m = distance_px * distance_per_pixel
    speed_mps = distance_m * fps
    return speed_mps * 3.6  # Convert to km/h

def detect_license_plate(image, bbox):
    """Enhanced license plate detection"""
    try:
        x1, y1, x2, y2 = bbox
        height, width = image.shape[:2]
        
        # Focus on lower part of vehicle for license plate
        plate_y1 = max(0, int(y2 - (y2-y1) * 0.3))
        plate_y2 = min(height, y2 + 10)
        plate_x1 = max(0, x1 - 5)
        plate_x2 = min(width, x2 + 5)
        
        plate_region = image[plate_y1:plate_y2, plate_x1:plate_x2]
        if plate_region.size == 0:
            return "Unknown"
            
        # Preprocess for better OCR
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Try OCR on both versions
        results_gray = reader.readtext(gray, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        results_thresh = reader.readtext(thresh, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        all_results = results_gray + results_thresh
        if all_results:
            all_results.sort(key=lambda x: x[2], reverse=True)
            plate_text = ''.join(e for e in all_results[0][1] if e.isalnum())
            return plate_text if len(plate_text) >= 2 else "Unknown"
    except Exception as e:
        logger.error(f"License plate detection error: {e}")
    
    return "Unknown"

def send_violation_notification(camera_name, vehicle_type, plate_number, speed, frame):
    """Send violation notification with images"""
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create violation directory
        os.makedirs("violations", exist_ok=True)
        
        # Save main image
        img_filename = f"violations/violation_{int(time.time())}.jpg"
        cv2.imwrite(img_filename, frame)
        
        # Prepare message
        message = (f"⚠️ PERINGATAN KECEPATAN BERLEBIH! ⚠️\n\n"
                  f"📹 Kamera: {camera_name}\n"
                  f"🚗 Kendaraan: {vehicle_type.title()}\n"
                  f"🔢 Plat: {plate_number}\n"
                  f"⚡ Kecepatan: {speed:.1f} km/jam\n"
                  f"⏰ Waktu: {current_time}\n\n"
                  f"Batas kecepatan: {SPEED_THRESHOLD} km/jam")
        
        # Send notification
        with open(img_filename, 'rb') as photo:
            bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=message)
        
        # Update stats
        system_stats["total_violations"] += 1
        system_stats["cameras"][camera_name]["violations"] += 1
        
        log_activity(f"Violation: {plate_number} at {speed:.1f} km/h on {camera_name}")
        
        # Cleanup after 5 minutes
        def cleanup():
            time.sleep(300)
            if os.path.exists(img_filename):
                os.remove(img_filename)
        threading.Thread(target=cleanup, daemon=True).start()
        
    except Exception as e:
        logger.error(f"Notification error: {e}")

def send_activity_report():
    """Send system activity report"""
    try:
        uptime = datetime.now() - system_stats["start_time"]
        uptime_str = str(uptime).split('.')[0]
        
        message = (f"📊 LAPORAN SISTEM PEMANTAUAN 📊\n\n"
                  f"⏱️ Uptime: {uptime_str}\n"
                  f"🚗 Total Kendaraan: {system_stats['total_vehicles']}\n"
                  f"🚨 Total Pelanggaran: {system_stats['total_violations']}\n\n"
                  f"📷 STATUS KAMERA:\n")
        
        for cam_name, stats in system_stats["cameras"].items():
            status = "✅" if stats["active"] else "❌"
            message += (f"{status} {cam_name}\n"
                       f"   Kendaraan: {stats['vehicles']}\n"
                       f"   Pelanggaran: {stats['violations']}\n")
        
        message += f"\n⚙️ Batas Kecepatan: {SPEED_THRESHOLD} km/jam"
        
        bot.send_message(chat_id=CHAT_ID, text=message)
        system_stats["last_activity_report"] = time.time()
        
    except Exception as e:
        logger.error(f"Activity report error: {e}")

def track_objects(prev_positions, new_positions, threshold=50):
    """Simple object tracking based on distance"""
    matched = {}
    used_new = set()
    
    for prev_id, prev_data in prev_positions.items():
        best_match = None
        best_distance = threshold
        
        for new_id, new_data in new_positions.items():
            if new_id in used_new:
                continue
                
            distance = np.sqrt((prev_data['center'][0] - new_data['center'][0])**2 + 
                             (prev_data['center'][1] - new_data['center'][1])**2)
            
            if distance < best_distance:
                best_distance = distance
                best_match = new_id
        
        if best_match is not None:
            matched[prev_id] = new_positions[best_match]
            used_new.add(best_match)
    
    # Add unmatched new detections with new IDs
    next_id = max(prev_positions.keys(), default=0) + 1
    for new_id, new_data in new_positions.items():
        if new_id not in used_new:
            matched[next_id] = new_data
            next_id += 1
    
    return matched

def process_camera(camera_index, window_name):
    """Main camera processing function"""
    global system_running
    
    log_activity(f"Starting {window_name}", send_telegram=True)
    
    # Initialize camera stats
    system_stats["cameras"][window_name] = {
        "active": False, "vehicles": 0, "violations": 0, "last_frame": time.time()
    }
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log_activity(f"Error: Cannot open camera {camera_index}", send_telegram=True)
        return
    
    system_stats["cameras"][window_name]["active"] = True
    prev_positions = {}
    frame_count = 0
    
    while cap.isOpened() and system_running:
        ret, frame = cap.read()
        if not ret:
            system_stats["cameras"][window_name]["active"] = False
            log_activity(f"{window_name} disconnected", send_telegram=True)
            break
        
        frame_count += 1
        current_time = time.time()
        system_stats["cameras"][window_name]["last_frame"] = current_time
        
        # Process every 2nd frame for performance
        if frame_count % 2 == 0:
            # YOLO detection
            results = model(frame)
            detections = results.pandas().xyxy[0]
            
            new_positions = {}
            vehicles_count = 0
            
            for idx, detection in detections.iterrows():
                if detection['name'] in TRACKED_VEHICLES and detection['confidence'] > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Detect license plate
                    plate_number = detect_license_plate(frame, (x1, y1, x2, y2))
                    
                    new_positions[idx] = {
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'class': detection['name'],
                        'plate': plate_number
                    }
                    
                    vehicles_count += 1
                    
                    # Draw bounding box and info
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{detection['name'].title()} - {plate_number}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Update vehicle count
            if vehicles_count > 0:
                system_stats["total_vehicles"] += vehicles_count
                system_stats["cameras"][window_name]["vehicles"] += vehicles_count
            
            # Track objects and calculate speeds
            tracked_objects = track_objects(prev_positions, new_positions)
            
            for obj_id, data in tracked_objects.items():
                if obj_id in prev_positions:
                    prev_center = prev_positions[obj_id]['center']
                    curr_center = data['center']
                    
                    speed = calculate_speed(prev_center, curr_center, FPS, DISTANCE_PER_PIXEL)
                    x1, y1, x2, y2 = data['bbox']
                    
                    # Display speed
                    color = (0, 0, 255) if speed > SPEED_THRESHOLD else (255, 255, 255)
                    cv2.putText(frame, f"{speed:.1f} km/h", (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Check for violations
                    if speed > SPEED_THRESHOLD:
                        plate = data['plate']
                        if (plate not in last_notification or 
                            current_time - last_notification.get(plate, 0) > NOTIFICATION_COOLDOWN):
                            
                            # Highlight violation
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, f"SPEEDING: {speed:.1f}", (x1, y1-40), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            
                            # Send notification
                            threading.Thread(
                                target=send_violation_notification,
                                args=(window_name, data['class'], plate, speed, frame.copy()),
                                daemon=True
                            ).start()
                            
                            last_notification[plate] = current_time
            
            prev_positions = tracked_objects
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        # Send periodic reports
        if current_time - system_stats["last_activity_report"] > ACTIVITY_LOG_INTERVAL:
            threading.Thread(target=send_activity_report, daemon=True).start()
        
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
    
    # Cleanup
    system_stats["cameras"][window_name]["active"] = False
    log_activity(f"{window_name} stopped", send_telegram=True)
    cap.release()
    cv2.destroyWindow(window_name)

def monitor_system():
    """Monitor system health"""
    global system_running
    while system_running:
        try:
            current_time = time.time()
            for cam_name, stats in system_stats["cameras"].items():
                if stats["active"] and current_time - stats["last_frame"] > 10:
                    stats["active"] = False
                    log_activity(f"⚠️ {cam_name} not responding", send_telegram=True)
            time.sleep(5)
        except Exception as e:
            logger.error(f"Health monitor error: {e}")

def run_telegram_bot():
    """Run Telegram bot"""
    try:
        log_activity("Telegram bot started")
        bot.polling(none_stop=True)
    except Exception as e:
        logger.error(f"Bot error: {e}")

# Telegram Bot Commands
@bot.message_handler(commands=['start'])
def cmd_start(message):
    global CHAT_ID
    CHAT_ID = str(message.chat.id)
    bot.reply_to(message, f"🚦 Speed Detection System Started!\nChat ID: {CHAT_ID}")
    log_activity(f"Bot started by {CHAT_ID}")

@bot.message_handler(commands=['status'])
def cmd_status(message):
    send_activity_report()

@bot.message_handler(commands=['threshold'])
def cmd_threshold(message):
    global SPEED_THRESHOLD
    try:
        new_threshold = float(message.text.split()[1])
        if new_threshold > 0:
            old_threshold = SPEED_THRESHOLD
            SPEED_THRESHOLD = new_threshold
            response = f"Speed limit changed: {old_threshold} → {SPEED_THRESHOLD} km/h"
            bot.reply_to(message, response)
            log_activity(response)
        else:
            bot.reply_to(message, "Speed limit must be > 0")
    except (IndexError, ValueError):
        bot.reply_to(message, "Usage: /threshold [speed_in_kmh]")

@bot.message_handler(commands=['report'])
def cmd_report(message):
    send_activity_report()

@bot.message_handler(commands=['log'])
def cmd_log(message):
    try:
        with open("speed_detection.log", 'rb') as log_file:
            bot.send_document(message.chat.id, log_file, caption="System Log File")
    except Exception as e:
        bot.reply_to(message, f"Error sending log: {e}")

@bot.message_handler(commands=['shutdown'])
def cmd_shutdown(message):
    global system_running
    bot.reply_to(message, "🔴 Shutting down system...")
    log_activity("Remote shutdown initiated", send_telegram=True)
    system_running = False
    threading.Thread(target=lambda: (time.sleep(2), os._exit(0)), daemon=True).start()

@bot.message_handler(commands=['help'])
def cmd_help(message):
    help_text = """🚦 SPEED DETECTION SYSTEM 🚦

Commands:
/start - Initialize bot
/status - System status
/report - Activity report  
/threshold [value] - Set speed limit
/log - Get log file
/shutdown - Remote shutdown
/help - This help message

Current speed limit: {SPEED_THRESHOLD} km/h"""
    bot.send_message(message.chat.id, help_text)

# Main execution
if __name__ == "__main__":
    try:
        # Start Telegram bot
        bot_thread = threading.Thread(target=run_telegram_bot, daemon=True)
        bot_thread.start()
        
        # Start system monitor
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
        
        # Send startup message
        log_activity("✅ Speed Detection System Started", send_telegram=True)
        
        # Start cameras
        camera_threads = []
        for i, name in enumerate(["Camera-1", "Camera-2"]):
            thread = threading.Thread(target=process_camera, args=(i, name))
            thread.start()
            camera_threads.append(thread)
        
        # Wait for camera threads
        for thread in camera_threads:
            thread.join()
            
    except KeyboardInterrupt:
        system_running = False
        log_activity("System stopped by user", send_telegram=True)
    except Exception as e:
        system_running = False
        log_activity(f"System error: {e}", send_telegram=True)
    finally:
        send_activity_report()
        log_activity("System shutdown complete", send_telegram=True)
        cv2.destroyAllWindows()