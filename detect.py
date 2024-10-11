import cv2
import mediapipe as mp
from ultralytics import YOLO
import pyautogui

model = YOLO('yolov8n.pt') # Load model YOLOV8 for object detection task

# Load media pose for making keypoint
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

screen_width, screen_height = pyautogui.size() # Get screen size by pyautogui


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA): # ปรับขนาดของรูปภาพให้ได้สัดส่วนคงเดิม
    (h, w) = image.shape[:2] #ดึงความกว้างและความสูงของรูปภาพต้นฉบับ
    if width is None and height is None: #ถ้าไม่ได้กำหนดขนาดของความสูงและความกว้าง ฟังก์ชั่นจะคืนค่าภาพต้นฉบับทันที
        return image
    if width is None: #ส่งมาเพียงแค่height
        r = height / float(h) #คำนวณหาอัตราส่วนในการปรับภาพ
        dim = (int(w * r), height) #นำอัตราส่วนที่ได้ไปคำนวณกับความกว้างเดิม 
    else: #ส่งมาแค่weight (ทำแบบเดิมแค่เปลี่ยนจากheightเป็นweight)
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter) #ส่งค่าคืนกลับไป

# IoU calculation
def calculate_iou(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1 # สกัด position ของ แกนx และ y ของ bounding box person
    x1_2, y1_2, x2_2, y2_2 = bbox2 # สกัด position ของ แกนx และ y ของ bounding box chair

    
    x_max_left = max(x1_1, x1_2) #หาจุดตัดของขอบซ้าย
    y_max_top = max(y1_1, y1_2) #หาจุดตัดของขอบวา
    x_min_right = min(x2_1, x2_2) #หาจุดตัดของขอบบน
    y_min_bottom = min(y2_1, y2_2) #หาจุดตัดของขอบล่าง

    # คำนวณหาจุดของ ความกว้าง และ ความสูง ของพื้นที่ทับซ้อน
    inter_width = max(0, x_min_right - x_max_left)
    inter_height = max(0, y_min_bottom - y_max_top)

    # คำนวณหาขนาดของพื้นที่ที่ทับซ้อนกัน
    intersection_area = inter_width * inter_height

    # คำนวณหาจุดที่ สองกล่อง Unionกัน
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # คำนวณหาพื้นที่ที่สองกล่อง Union กัน
    union_area = area1 + area2 - intersection_area

    # คำรวณหาค่าIoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

# ฟังก์ชั่นในการตรวจสอบว่าuserนั่งอยู่บนเก้าอี้ไหม
def is_sit_chair(user_bbox, chair_bbox, threshold=0.35):
    iou = calculate_iou(user_bbox, chair_bbox) #คำนวณหาค่าIoUของ bbox_user และ bbox_chair
    return iou >= threshold

# ฟังก์ชั่นในการตรวจสอบว่าภาพที่userส่งมานั้นเห็นเต็มตัวหรือครึ่งตัว
def is_full_body(pose_landmarks):
    left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE] #สกัดเข่าซ้ายด้วยmedia pose
    right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE] #สกัดเข่าขวาด้วยmedia pose

    # ถ้าตรวจจับได้เข่าทั้งสองข้าง ถือว่าเป็น "Full Body"
    if left_knee.visibility > 0.5 and right_knee.visibility > 0.5:
        return True
    return False

# ฟังก์ชั่นในการดึงรูปภาพ
cap = cv2.VideoCapture(0)

person_bboxes = []  # เก็บ bounding box ของหลายบุคคล
chair_bbox = None   # เก็บ bounding box ของเก้าอี้

while cap.isOpened(): #เปิดลูปและทำงานเมื่อรู้ว่ากล้องหรือวิดีโอถูกเปิด
    ret, frame = cap.read() #อ่านฟรมทีละเฟรม
    if not ret: #ถ้าไม่สามารถอ่านเฟรมได้จะหยุดการทำงานทันที
        break

    # YOLO detection
    results = model(frame)
    
    person_bboxes.clear()

    # Loop through all detected objects
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) #ดึงพิกัดของbounding boxของแต่ละclassที่detectเจอ
            class_id = int(box.cls) #ดึงlcass_idของแต่ละอัน

            if class_id == 0 or class_id == 56: #0 คือ person 56 คือ chair
                # วาด bounding ของแต่ละclass
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # ใส่ชื่อclassและความแม่นยำลงด้านบนของbounging box
                class_name = model.names[class_id]
                confidence = box.conf[0] * 100
                label = f"{class_name} {confidence:.1f}%"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Person detection
                if class_id == 0: #ตรวจสอบว่า class เป็น personไหม
                    person_bboxes.append((x1, y1, x2, y2))  # เก็บ bounding box ของบุคคล
                    
                    person_crop = frame[y1:y2, x1:x2] #ตัดเฟรมออกมาเอาเฉพาะส่วนที่เป็นperson
                    person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB) #แปลงให้เป็นRGB เพื่อให้ใช่งานร่วมกับmediapipeได้

                    # Detect pose landmarks with MediaPipe
                    result = pose.process(person_rgb)
                    
                    if result.pose_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(person_crop, result.pose_landmarks, mp_pose.POSE_CONNECTIONS) # หากตรวจพบ keypoints จะแสดงผล keypoints เหล่านั้นบนเฟรมโดยใช้ draw_landmarks
                    
                    if is_full_body(result.pose_landmarks): #นำpose_landmarksที่เจอส่งเข้าไปในฟังก์ชั่นis_full_body
                        cv2.putText(frame, "Person is detected (Full Body)", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, "Person is detected (Half Body)", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 255), 2)
                
                # Chair detection
                if class_id == 56:
                    chair_bbox = (x1, y1, x2, y2)  # เก็บ bounding box ของเก้าอี้

    # ตรวจสอบจำนวนบุคคลที่ตรวจพบ
    if len(person_bboxes) > 1:
        # ถ้าตรวจพบมากกว่า 1 คน
        cv2.putText(frame, "can detect only one person", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    elif len(person_bboxes) == 1:
        person_bbox = person_bboxes[0]  # ใช้ bounding box ของคนเดียวที่พบ
        if chair_bbox:  # จะเริ่มทำการตรวจสอบเมื่อตรวจพบเก้าอี้
            if is_sit_chair(person_bbox, chair_bbox): #ส่งbounding box ของ personและ chair ไปให้ฟังก์ชั่น is_sit_chair
                cv2.putText(frame, "Person is sitting on the chair", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Person is NOT sitting on the chair", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Chair not detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 2)

    # Resize frame and display
    resized_frame = resize_with_aspect_ratio(frame, width=screen_width, height=screen_height)
    cv2.imshow('YOLO Object Detection', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #ลูปจะหยุดทำงานเมื่อผู้ใช้กดปุ่ม 'q'
        break

cap.release()
cv2.destroyAllWindows()
