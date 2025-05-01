import cv2
from datetime import datetime
import time
import threading
import numpy as np
import queue
import os
import json
import csv
from tensorflow.lite.python.interpreter import Interpreter
import easyocr
import logging

# Configuration constants
FPS = 10  # Reduced for Raspberry Pi performance
DISTANCE_PER_PIXEL = 0.00625  # 1 meter ≈ 160 pixels at 4m height
SPEED_THRESHOLD = 20  # km/h
CONFIDENCE_THRESHOLD = 0.5
CAMERA_SWITCH_INTERVAL = 5  # seconds for each camera
PROCESS_OCR_EVERY_N_FRAMES = 10  # Process OCR only every N frames
LOG_DIR = "detection_logs"  # Directory to store logs
CSV_LOG_FILE = os.path.join(LOG_DIR, f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
JSON_LOG_FILE = os.path.join(LOG_DIR, f"detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
IMAGE_LOG_DIR = os.path.join(LOG_DIR, "images")  # Directory to store detected object images

# Define class names
CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Initialize cameras
camera_stream = None
current_camera = 0
processing_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=5)  # Buffer a few frames
ocr_queue = queue.Queue()  # Queue for license plate recognition
results_dict = {}  # Store detection results

# Using TFLite instead of full YOLO for better performance on Raspberry Pi
def load_model():
    # MobileNet SSD is lighter than YOLO
    model_path = 'detect.tflite'  # Download this from TensorFlow model zoo
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_output_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    return input_details, output_details, height, width

def detect_objects(interpreter, image, threshold):
    input_details, output_details, height, width = get_output_details(interpreter)
    
    # Resize and normalize image
    image_resized = cv2.resize(image, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
    
    # If model expects float32 input
    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get outputs
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
    detections = []
    for i in range(len(scores)):
        if scores[i] > threshold and int(classes[i]) in CLASSES:
            # Denormalize bounding box coordinates
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * image.shape[1])
            xmax = int(xmax * image.shape[1])
            ymin = int(ymin * image.shape[0])
            ymax = int(ymax * image.shape[0])
            
            class_id = int(classes[i])
            detections.append({
                "bbox": (xmin, ymin, xmax, ymax),
                "class_id": class_id,
                "label": CLASSES.get(class_id, "Unknown"),
                "confidence": float(scores[i]),
                "center": ((xmin + xmax) // 2, (ymin + ymax) // 2)
            })
    
    return detections

def calculate_speed(prev_center, curr_center, fps, distance_per_pixel):
    distance_px = np.sqrt((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2)
    distance_m = distance_px * distance_per_pixel
    speed_mps = distance_m * fps
    speed_kmh = speed_mps * 3.6
    return speed_kmh

def ocr_worker():
    """Worker thread for license plate OCR processing"""
    # Initialize OCR here to avoid memory usage in main thread
    reader = easyocr.Reader(['en'], gpu=False)  # GPU=False for Pi compatibility
    
    while True:
        try:
            item = ocr_queue.get(timeout=1)
            if item is None:  # Sentinel value to exit
                break
                
            image, bbox, timestamp, object_id = item
            plate_text = "Unknown"
            
            try:
                x1, y1, x2, y2 = bbox
                # Adjust ROI for license plate (focus on lower part of vehicle)
                plate_roi_y1 = int(y1 + (y2-y1)*0.7)  # Take lower 30% of object
                plate_region = image[plate_roi_y1:y2, max(0, x1-5):min(image.shape[1], x2+5)]
                
                if plate_region.size > 0 and plate_region.shape[0] > 10 and plate_region.shape[1] > 10:
                    # Apply preprocessing for better OCR
                    gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Run OCR
                    results = reader.readtext(thresh)
                    if results:
                        plate_text = results[0][1]
                        # Filter for likely license plate format
                        # This could be adjusted for your country's plate format
                        if len(plate_text) >= 3:
                            # Store result
                            with processing_lock:
                                if object_id in results_dict:
                                    results_dict[object_id]["plate"] = plate_text
            except Exception as e:
                print(f"OCR error: {e}")
                
            ocr_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"OCR worker error: {e}")

def camera_switch_thread():
    """Thread to handle camera switching"""
    global camera_stream, current_camera
    
    while True:
        time.sleep(CAMERA_SWITCH_INTERVAL)
        with processing_lock:
            if camera_stream is not None:
                camera_stream.release()
            
            # Switch camera
            current_camera = 1 if current_camera == 0 else 0
            camera_stream = cv2.VideoCapture(current_camera)
            
            # Set lower resolution for better performance
            camera_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera_stream.set(cv2.CAP_PROP_FPS, FPS)
            
            logging.info(f"Switched to camera {current_camera}")
            
            # Clear results when switching cameras to avoid false matches
            with processing_lock:
                for k, v in results_dict.items():
                    if not v.get("logged", False) and v["label"] in ["Car", "Motorcycle", "Bus", "Truck"] and v["speed"] > 0:
                        log_detection(k, v, 1 if current_camera == 0 else 0)
                results_dict.clear()

def frame_grabber():
    """Thread to continuously grab frames from the active camera"""
    global camera_stream
    
    while True:
        with processing_lock:
            if camera_stream is None or not camera_stream.isOpened():
                time.sleep(0.1)
                continue
                
            ret, frame = camera_stream.read()
            
        if ret:
            # Don't block if queue is full, just drop frames
            try:
                frame_queue.put((frame, time.time()), block=False)
            except queue.Full:
                pass
        else:
            time.sleep(0.01)

def setup_logging():
    """Set up logging configuration"""
    # Create log directories if they don't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(IMAGE_LOG_DIR, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, f"system_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
            logging.StreamHandler()
        ]
    )
    
    # Initialize CSV log file with headers
    with open(CSV_LOG_FILE, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'Timestamp', 
            'Camera', 
            'Object_ID', 
            'Class', 
            'Speed (km/h)', 
            'License Plate', 
            'Confidence',
            'Position_X',
            'Position_Y'
        ])
    
    # Initialize JSON log with metadata
    with open(JSON_LOG_FILE, 'w') as jsonfile:
        json.dump({
            "metadata": {
                "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "fps": FPS,
                "distance_per_pixel": DISTANCE_PER_PIXEL,
                "speed_threshold": SPEED_THRESHOLD,
                "confidence_threshold": CONFIDENCE_THRESHOLD
            },
            "detections": []
        }, jsonfile)
    
    logging.info(f"Logging initialized. CSV log: {CSV_LOG_FILE}, JSON log: {JSON_LOG_FILE}")

def log_detection(object_id, detection_data, camera_id, frame=None):
    """Log detection data to CSV, JSON, and optionally save image"""
    timestamp = datetime.fromtimestamp(detection_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')
    label = detection_data['label']
    speed = detection_data['speed']
    plate = detection_data['plate']
    bbox = detection_data['bbox']
    confidence = detection_data.get('confidence', 0.0)
    position_x, position_y = detection_data.get('center', (0, 0))
    
    # Log to CSV file
    with open(CSV_LOG_FILE, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            timestamp,
            camera_id,
            object_id,
            label,
            f"{speed:.2f}",
            plate,
            f"{confidence:.2f}",
            position_x,
            position_y
        ])
    
    # Log to JSON file (append to detections array)
    detection_entry = {
        "id": object_id,
        "timestamp": timestamp,
        "camera": camera_id,
        "label": label,
        "speed": round(speed, 2),
        "license_plate": plate,
        "confidence": round(confidence, 2),
        "bounding_box": {
            "x1": bbox[0],
            "y1": bbox[1],
            "x2": bbox[2],
            "y2": bbox[3]
        },
        "position": {
            "x": position_x,
            "y": position_y
        }
    }
    
    # Append to JSON file
    with open(JSON_LOG_FILE, 'r+') as jsonfile:
        data = json.load(jsonfile)
        data["detections"].append(detection_entry)
        jsonfile.seek(0)
        json.dump(data, jsonfile, indent=2)
        jsonfile.truncate()
    
    # Save image clip if frame is provided and it's a vehicle with speed over threshold
    if frame is not None and speed > SPEED_THRESHOLD and label in ["Car", "Motorcycle", "Bus", "Truck"]:
        try:
            x1, y1, x2, y2 = bbox
            # Add some padding
            pad = 20
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(frame.shape[1], x2+pad), min(frame.shape[0], y2+pad)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                image_filename = os.path.join(
                    IMAGE_LOG_DIR, 
                    f"cam{camera_id}_{object_id.replace('obj_', '')}_{label}_{speed:.1f}kmh.jpg"
                )
                cv2.imwrite(image_filename, crop)
                logging.info(f"Saved detection image: {image_filename}")
        except Exception as e:
            logging.error(f"Error saving detection image: {e}")

def main():
    global camera_stream, current_camera
    
    # Setup logging
    setup_logging()
    logging.info("Starting traffic monitoring system")
    
    # Initialize first camera
    camera_stream = cv2.VideoCapture(current_camera)
    camera_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera_stream.set(cv2.CAP_PROP_FPS, FPS)
    
    logging.info(f"Camera {current_camera} initialized")
    
    logging.info("Loading TFLite model...")
    interpreter = load_model()
    logging.info("Model loaded successfully!")
    
    # Start camera switching thread
    switch_thread = threading.Thread(target=camera_switch_thread, daemon=True)
    switch_thread.start()
    
    # Start frame grabber thread
    grabber_thread = threading.Thread(target=frame_grabber, daemon=True)
    grabber_thread.start()
    
    # Start OCR worker thread
    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    ocr_thread.start()
    
    prev_positions = {}
    frame_count = 0
    
    try:
        while True:
            try:
                frame, timestamp = frame_queue.get(timeout=1)
                frame_count += 1
                
                # Process frame
                detections = detect_objects(interpreter, frame, CONFIDENCE_THRESHOLD)
                new_positions = {}
                
                # Process detections
                for i, detection in enumerate(detections):
                    object_id = f"obj_{timestamp}_{i}"
                    bbox = detection["bbox"]
                    label = detection["label"]
                    center = detection["center"]
                    x1, y1, x2, y2 = bbox
                    
                    new_positions[object_id] = {
                        "bbox": bbox,
                        "label": label,
                        "center": center,
                        "timestamp": timestamp
                    }
                    
                    # Add to results dictionary
                    results_dict[object_id] = {
                        "bbox": bbox,
                        "label": label,
                        "timestamp": timestamp,
                        "speed": 0,
                        "plate": "Unknown"
                    }
                    
                    # Calculate speed for vehicles
                    if label in ["Car", "Motorcycle", "Bus", "Truck"]:
                        # Find closest match in previous positions for speed calculation
                        min_dist = float('inf')
                        matched_id = None
                        
                        for prev_id, prev_data in prev_positions.items():
                            if prev_data["label"] == label:
                                prev_center = prev_data["center"]
                                dist = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                                if dist < min_dist and dist < 100:  # Max distance threshold
                                    min_dist = dist
                                    matched_id = prev_id
                        
                        # Calculate speed if match found
                        if matched_id:
                            prev_center = prev_positions[matched_id]["center"]
                            prev_time = prev_positions[matched_id]["timestamp"]
                            time_diff = timestamp - prev_time
                            
                            if time_diff > 0:
                                actual_fps = 1.0 / time_diff
                                speed = calculate_speed(prev_center, center, actual_fps, DISTANCE_PER_PIXEL)
                                results_dict[object_id]["speed"] = speed
                        
                        # Queue for OCR processing every N frames
                        if frame_count % PROCESS_OCR_EVERY_N_FRAMES == 0:
                            try:
                                ocr_queue.put((frame.copy(), bbox, timestamp, object_id), block=False)
                            except queue.Full:
                                pass
                
                # Visualize results
                for obj_id, data in results_dict.items():
                    if timestamp - data["timestamp"] < 2.0:  # Only show recent detections
                        bbox = data["bbox"]
                        label = data["label"]
                        speed = data["speed"]
                        plate = data["plate"]
                        
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw labels with speed and plate
                        if label in ["Car", "Motorcycle", "Bus", "Truck"]:
                            color = (0, 0, 255) if speed > SPEED_THRESHOLD else (255, 255, 255)
                            text = f"{label}: {speed:.1f} km/h"
                            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            if plate != "Unknown":
                                cv2.putText(frame, f"Plate: {plate}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        else:
                            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw camera info
                cv2.putText(frame, f"Camera: {current_camera}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Clean old entries from results_dict
                current_time = time.time()
                keys_to_remove = [k for k, v in results_dict.items() if current_time - v["timestamp"] > 10.0]
                
                # Log completed detections before removing them
                for k in keys_to_remove:
                    # Only log detections that have been fully processed
                    if results_dict[k].get("logged", False) == False:
                        # Mark as logged to prevent duplicate logs
                        results_dict[k]["logged"] = True
                        
                        # For vehicles with a detected speed, log the final result
                        if results_dict[k]["label"] in ["Car", "Motorcycle", "Bus", "Truck"] and results_dict[k]["speed"] > 0:
                            log_detection(k, results_dict[k], current_camera, frame.copy())
                            logging.info(f"Logged detection: {results_dict[k]['label']} with speed {results_dict[k]['speed']:.1f} km/h and plate {results_dict[k]['plate']}")
                
                # Remove old entries after logging
                for k in keys_to_remove:
                    results_dict.pop(k, None)
                
                # Show the frame
                cv2.imshow("Traffic Monitor", frame)
                
                # Update previous positions
                prev_positions = new_positions
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except queue.Empty:
                continue
                
    except KeyboardInterrupt:
        logging.info("User interrupted. Shutting down...")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        # Final log summary
        logging.info("Creating final log summary...")
        
        # Log any remaining unlogged detections
        for obj_id, data in results_dict.items():
            if not data.get("logged", False) and data["label"] in ["Car", "Motorcycle", "Bus", "Truck"] and data["speed"] > 0:
                log_detection(obj_id, data, current_camera)
        
        # Generate summary statistics
        try:
            with open(JSON_LOG_FILE, 'r') as jsonfile:
                data = json.load(jsonfile)
                
                # Add summary data
                detection_count = len(data["detections"])
                vehicle_count = sum(1 for d in data["detections"] if d["label"] in ["Car", "Motorcycle", "Bus", "Truck"])
                speeding_count = sum(1 for d in data["detections"] if d["speed"] > SPEED_THRESHOLD)
                
                data["summary"] = {
                    "total_detections": detection_count,
                    "vehicle_count": vehicle_count,
                    "speeding_count": speeding_count,
                    "speeding_percentage": round((speeding_count / vehicle_count * 100) if vehicle_count > 0 else 0, 2),
                    "end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "duration_seconds": round((datetime.now() - datetime.strptime(data["metadata"]["start_time"], '%Y-%m-%d %H:%M:%S')).total_seconds(), 2)
                }
                
                # Write updated data back to file
                with open(JSON_LOG_FILE, 'w') as outfile:
                    json.dump(data, outfile, indent=2)
                
                logging.info(f"Summary: {detection_count} total detections, {vehicle_count} vehicles, {speeding_count} speeding")
        except Exception as e:
            logging.error(f"Error creating summary statistics: {e}")
        
        # Cleanup
        logging.info("Cleaning up and exiting...")
        ocr_queue.put(None)  # Signal OCR thread to exit
        
        if ocr_thread.is_alive():
            ocr_thread.join(timeout=1)
            
        if camera_stream is not None:
            camera_stream.release()
            
        cv2.destroyAllWindows()
        
        logging.info(f"Processing complete. Logs available at: {LOG_DIR}")
        
        # Create a README file with instructions for viewing logs
        readme_path = os.path.join(LOG_DIR, "README.txt")
        with open(readme_path, 'w') as readme:
            readme.write(f"""Detection Logs - Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Log Files:
- CSV Report: {os.path.basename(CSV_LOG_FILE)}
- JSON Data: {os.path.basename(JSON_LOG_FILE)}
- System Log: system_log_*.log
- Detection Images: ./images/ directory

How to use these logs:
1. The CSV file can be opened with any spreadsheet software for analysis
2. The JSON file contains complete detection data with metadata
3. The system log contains runtime information
4. Images of vehicles exceeding {SPEED_THRESHOLD} km/h are stored in the images directory

To generate a simple summary report:
$ python -c "import json; f=open('{JSON_LOG_FILE}'); data=json.load(f); print(json.dumps(data['summary'], indent=2))"
""")

if __name__ == "__main__":
    main()
