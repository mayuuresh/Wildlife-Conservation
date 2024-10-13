from ultralytics import YOLO
import cv2
import math
import time

# This code is from chatgpt to access camera from mobile
    # Replace with your mobile camera's IP address and port number
    # mobile_ip = "192.168.1.4"  # Example IP address
    # port = "8080"              # Example port number
    # url = f"http://{mobile_ip}:{port}/video"
    # Initialize video capture from the mobile camera
    
cap = cv2.VideoCapture(0)

# Load YOLO model
model = YOLO("./best.pt")

last_print_time = time.time()
tracked_elephants = []

bbox_threshold = 50  

def is_new_elephant(x1, y1, x2, y2, tracked_elephants):
    """Check if the elephant is new by comparing the bounding box with tracked elephants."""
    for elephant in tracked_elephants:
        px1, py1, px2, py2 = elephant
        if abs(x1 - px1) < bbox_threshold and abs(y1 - py1) < bbox_threshold:
            return False  
    return True  

elephant_count = 0

button_width = 100
button_height = 50
button_x = 540  
button_y = 420
button_color = (0, 0, 255)  
button_text = "Quit"

terminate_program = False

def on_mouse_click(event, x, y, flags, param):
    global terminate_program
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height:
            terminate_program = True

cv2.namedWindow('Webcam')
cv2.setMouseCallback('Webcam', on_mouse_click)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    results = model(img, stream=True)

    current_time = time.time()

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])

            if cls == 0:  
                if is_new_elephant(x1, y1, x2, y2, tracked_elephants):
                    elephant_count += 1  
                    tracked_elephants.append((x1, y1, x2, y2))  

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, f'Elephant {confidence}', org, font, fontScale, color, thickness)

    if current_time - last_print_time >= 1:
        print(f"Number of elephants detected: {elephant_count}")
        last_print_time = current_time  

    cv2.putText(img, f'Elephants Count: {elephant_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.rectangle(img, (button_x, button_y), (button_x + button_width, button_y + button_height), button_color, -1)
    cv2.putText(img, button_text, (button_x + 10, button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)

    if terminate_program or cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
