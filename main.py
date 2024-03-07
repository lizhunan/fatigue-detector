import cv2
from dlib import get_frontal_face_detector, shape_predictor
import numpy as np

ear_threshold = 0.2
mar_threshold = 0.6
eye_counter = 0
eye_total = 0
mouth_counter = 0
mouth_total = 0
blinks_threshold = 12
yawning_threshold = 6

def init(ear, mar, max_blinks, max_yawning):
    global ear_threshold
    global mar_threshold
    global blinks_threshold
    global yawning_threshold
    ear_threshold = ear
    mar_threshold = mar
    blinks_threshold = max_blinks
    yawning_threshold = max_yawning
    detector = get_frontal_face_detector()
    predictor = shape_predictor('model/shape_predictor_68_face_landmarks.dat')
    return detector, predictor

def get_landmarks(image, detector, predictor):
    gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_im, 1)
    if len(dets) == 0:
        return None, None
    shape = predictor(gray_im, dets[0])
    return np.array([[part.x, part.y] for part in shape.parts()]), dets[0]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) /(2.0 * C)

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    return (A + B) / (2.0 * C)

def fatigue_detector(image, detector, predictor):

    global eye_counter
    global eye_total
    global mouth_counter
    global mouth_total

    landmarks, dets = get_landmarks(image, detector, predictor)
    if landmarks is None:
        return image
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    mouth = landmarks[48:68]
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    mar = mouth_aspect_ratio(mouth)
    ear = (right_ear + left_ear) / 2.0

    if ear < ear_threshold:
        eye_counter += 1
    else:
        if eye_counter >= 5:
            eye_total += 1
        eye_counter = 0
    
    if mar > mar_threshold:
        mouth_counter += 1
    else:
        if mouth_counter >= 5:
            mouth_total += 1
        mouth_counter = 0

    cv2.putText(image,'BLINKS:{}'.format(eye_total),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.putText(image,'EAR-COUNTER:{}'.format(eye_counter), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image,'EAR:{:.2f}'.format(ear), (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(image, 'YAWNING:{}'.format(mouth_total), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, 'MAR-COUNTER:{}'.format(mouth_counter), (150,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, 'MAR:{:.2f}'.format(mar), (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if eye_total >= blinks_threshold or mouth_total >= yawning_threshold:
        cv2.putText(image, 'FATIGUE',(dets.right(), dets.bottom()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.rectangle(image, (dets.left(), dets.top()), (dets.right(), dets.bottom()), (0, 0, 255), 1)
    else:
        cv2.rectangle(image, (dets.left(), dets.top()), (dets.right(), dets.bottom()), (0, 255, 0), 1)
    
    return image