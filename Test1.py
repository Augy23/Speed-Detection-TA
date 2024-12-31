import cv2
from datetime import datetime
import telegram
from ultralytics import YOLO
import numpy as np

# Inisialisasi model YOLO
model = YOLO("yolov8n.pt")

# Konfigurasi Telegram
BOT_TOKEN = "7866793055:AAFg6KnnyNVo3A6L5MS65GjpeReKa8Ws5q0"  # Token bot Anda
CHAT_ID = "1948731720"  # Ganti dengan chat ID yang sesuai
bot = telegram.Bot(token=BOT_TOKEN)

# Parameter untuk deteksi
FPS = 30  # Frame rate video
DISTANCE_BETWEEN_FRAMES = 10  # Jarak fisik antar frame dalam meter
SPEED_THRESHOLD = 0.5  # Kecepatan (km/j) lebih rendah untuk pengujian

# Fungsi untuk menghitung kecepatan
def calculate_speed(prev_position, curr_position, fps, distance_between_frames):
    distance_px = ((curr_position[0] - prev_position[0]) ** 2 + (curr_position[1] - prev_position[1]) ** 2) ** 0.5
    distance_m = distance_px * distance_between_frames / 100  # Kalibrasi sederhana
    time_s = 1 / fps
    speed_kmh = (distance_m / time_s) * 3.6
    return speed_kmh

# Fungsi untuk mengirim notifikasi ke Telegram
def send_telegram_notification(image_path, speed, street_name, timestamp):
    try:
        with open(image_path, 'rb') as photo:
            message = f"""
🚗 **Speed Violation Detected!**
- Speed: {speed:.2f} km/h
- Street: {street_name}
- Time: {timestamp}
"""
            bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=message, parse_mode="Markdown")
            print("Notifikasi terkirim ke Telegram.")
    except Exception as e:
        print(f"Error saat mengirim pesan ke Telegram: {e}")

# Pipeline utama
cap = cv2.VideoCapture(0)  # Menggunakan kamera laptop (0 untuk default camera)
street_name = "Complex Street A"
prev_positions = {}
previous_frame = None
motion_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if previous_frame is None:
        previous_frame = gray_frame
        continue

    frame_diff = cv2.absdiff(previous_frame, gray_frame)
    _, thresh_frame = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Gambarkan kotak hijau

    if motion_detected:
        results = model(frame)
        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                if conf > 0.3:  # Ambang batas confidence untuk pengujian
                    x1, y1, x2, y2 = map(int, box)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    object_id = hash(tuple(center))
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Simpan screenshot
                    screenshot_path = f"screenshot_{timestamp.replace(':', '-')}.jpg"
                    cv2.imwrite(screenshot_path, frame)

                    # Kirim notifikasi ke Telegram
                    try:
                        message = f"""
🚗 **Object Detected!**
- Street: {street_name}
- Time: {timestamp}
"""
                        with open(screenshot_path, 'rb') as photo:
                            bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=message, parse_mode="Markdown")
                        print("Notifikasi terkirim ke Telegram.")
                    except Exception as e:
                        print(f"Error saat mengirim gambar ke Telegram: {e}")

                    prev_positions[object_id] = center

                    # Hitung kecepatan jika objek sudah ada sebelumnya
                    if object_id in prev_positions:
                        speed = calculate_speed(prev_positions[object_id], center, FPS, DISTANCE_BETWEEN_FRAMES)

                        # Kirim notifikasi jika kecepatan melebihi threshold
                        if speed > SPEED_THRESHOLD:
                            send_telegram_notification(screenshot_path, speed, street_name, timestamp)

    previous_frame = gray_frame
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
