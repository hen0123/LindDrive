import cv2
import mediapipe as mp
import torch
import yaml
import numpy as np
from tqdm import tqdm
import seaborn as sn
import ultralytics
import datetime
import math
import os

# 각도 측정

def calculate_angle(a,b,c): # sholuder , elbow, wrist
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radian= np.abs((np.arctan2(c[1]-b[1], c[0]-b[0])*180 / np.pi) - (np.arctan2(a[1]-b[1], a[0]-b[0])*180/np.pi))

    if radian > 180.0:
        radian = 360 - radian

    print(radian)
    
    return radian

# 점수측정
def check(is_count):
    score = 0
    part = [20,20,20,20,20]
    for i in range(5):
        part[i] -= (is_count[i] * 0.1)
    score = sum(part)

    return score

# return값을 설정하지 않음
def posture_video(video,path):
    data = {'comment1':"좋은 자세입니다.",'comment2':"좋은 자세입니다.",'comment3':"좋은 자세입니다.",'comment4':"좋은 자세입니다.","TackBack":"c:없음","Swing":"c:없음","Impact":"c:없음","Followthrough":"c:없음"}
    tb_angle = [0,0]
    s_angle = [0,0]
    i_angle = [0,0]
    f_angle = [0,0]

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    video = video
    path = path
    upload_path = os.path.join(path, video)
    load = os.path.join('C:/Users/Hyuni/Desktop/Dacapo/LineDrive/code/video/', video)
    pic = load.split('.')[0]
    cap = cv2.VideoCapture(upload_path)

    # 저장기능
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')    
    out = cv2.VideoWriter(load+"_result.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2) # 모델 복잡도 => 값이 클수록 복잡합(정확)

    # score
    # 교정 카운트
    is_count = [0,0,0,0,0]
    # 사진 촬영유무
    is_capture = [True,True,True,True]
    # 어드레스 시 첫 프레임을 받아오기 위한 플래그
    is_first = True
    # 어드레스 시 첫 프레임의 좌표를 저장할 변수 => 얼굴위치 및 크기 설정
    first_center_x, first_center_y, first_radius = None, None, None
    # 점수 측정
    score = 0
    # yolo 모델 생성
    model = torch.hub.load("ultralytics/yolov5", "custom", "code/best.pt", force_reload=True )

    while cap.isOpened():
        ret, img = cap.read()
        yolo = model(img)
        if not ret:
            break

        img_h, img_w, _ = img.shape

        img_result = img.copy()   
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img)
        skeleton = {(12, 14),(14,16),(12, 24),(23, 24),(23, 25),(24, 26),(25, 27),(26, 28),(27, 29), (27, 31),(28, 30),(28, 32),(29, 31),(30, 32)}
        mp_drawing.draw_landmarks(
            img_result,
            results.pose_landmarks,
            skeleton, # mp_pose.POSE_CONNECTIONS
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # 만든 로직
        if results.pose_landmarks:
            landmark = results.pose_landmarks.landmark
            
            # landmark[mp_pose.PoseLandmark.LEFT_EAR].x => % 값이므로 사진의 너비를 곱해줘서 px값으로 변환
            left_ear_x = landmark[mp_pose.PoseLandmark.LEFT_EAR].x * img_w
            left_ear_y = landmark[mp_pose.PoseLandmark.LEFT_EAR].y * img_h

            right_ear_x = landmark[mp_pose.PoseLandmark.RIGHT_EAR].x * img_w
            right_ear_y = landmark[mp_pose.PoseLandmark.RIGHT_EAR].y * img_h

            right_foot_x = landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * img_w
            right_foot_y = landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * img_h
            
            center_x = int((left_ear_x + right_ear_x) / 2)
            center_y = int((left_ear_y + right_ear_y) / 2)

            radius = int((left_ear_x - right_ear_x) / 2)
            radius = max(radius, 20)

            shoulder = [landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x*img_w ,landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y*img_h]
            elbow = [landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x*img_w ,landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y*img_h]
            wrist = [landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x*img_w ,landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y*img_h]

            angle = calculate_angle(shoulder,elbow,wrist)

            # 어드레스 시 첫 프레임의 머리 좌표 저장
            if is_first == False:
                cv2.circle(img_result, center=(first_center_x, first_center_y),
                    radius=first_radius, color=(0, 255, 255), thickness=2)

                color = (0, 255, 0) # 초록색

                # 머리가 원래 위치보다 많이 벗어난 경우
                if center_x - radius < first_center_x - first_radius \
                    or center_x + radius > first_center_x + first_radius:
                    color = (0, 0, 255) # 빨간색
                    is_count[0] +=1

                cv2.circle(img_result, center=(center_x, center_y),
                    radius=radius, color=color, thickness=2)
            
            

        # 자세 감지 
        for obj in yolo.xyxy[0]:  
            if obj[4] > 0.8:
                
                text_x = right_foot_x
                text_y = img_h - radius
                class_label = yolo.names[int(obj[5])]
                cv2.putText(img_result, f'{class_label} : {str(round(angle,3))}', (int(text_x),int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) 
                img_path = f'{pic}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
                
                if int(obj[5]) == 1: # 테이크백
                    tb_angle[0] += angle
                    tb_angle[1] +=1
                    if is_capture[0]:
                        print(is_capture[0])
                        is_capture[0] = False
                        cv2.imwrite(img_path+"TackBack.jpg", img_result)
                        data['TackBack'] = img_path+"TackBack.jpg"
                    
                    if angle > 45:
                        is_count[1] += 1
                
                elif int(obj[5]) == 3: # 스윙
                    s_angle[0] += angle
                    s_angle[1] +=1
                    if is_first: 
                        first_center_x = center_x
                        first_center_y = center_y
                        first_radius = int(radius *2.2)
                        is_first = False
                    
                    cv2.circle(img_result, center=(first_center_x, first_center_y),
                    radius=first_radius, color=(0, 255, 255), thickness=2)

                    if angle < 65 or angle > 75:
                        is_count[2] += 1

                    if is_capture[1]:
                        is_capture[1] = False
                        cv2.imwrite(img_path+"Swing.jpg", img_result)
                        data['Swing'] = img_path +"Swing.jpg"

                
                elif int(obj[5]) == 0: # 임팩트
                    i_angle[0] += angle
                    i_angle[1] +=1
                    if is_capture[2]:
                        is_capture[2] = False
                        cv2.imwrite(img_path +"Impact.jpg", img_result)
                        data['Impact'] = img_path +"Impact.jpg"
                    if angle < 85 or angle > 100:
                        is_count[3] += 1
                
                elif int(obj[5]) == 2: # 팔로윙스로우
                    f_angle[0] += angle
                    f_angle[1] +=1
                    if is_capture[3]:
                        is_capture[3] = False
                        cv2.imwrite(img_path +"Followthrough.jpg", img_result)
                        data['Followthrough'] = img_path +"Followthrough.jpg"

                        if angle < 120 :
                            is_count[4] += 1
                        score = check(is_count)
                        data['score'] = str(int(score))
                        print(score)   
                        print(f'swing: {s_angle[0]/s_angle[1]} | impact: {i_angle[0]/i_angle[1]} | followthrough: {f_angle[0]/f_angle[1]}')        
                        cap.release()
                        out.release()
                        cv2.destroyAllWindows()
                    

        
        cv2.imshow('LineDrive', img_result) 
        out.write(img_result)

        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return data
