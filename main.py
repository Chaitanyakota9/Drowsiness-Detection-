from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
from playsound import playsound
import pandas as pd
import keras
from twilio.rest import Client

Detection = dlib.get_frontal_face_detector()
Prediction = dlib.shape_predictor("/Users/chaitanyakota/Downloads/drowsiness detection/shape_predictor_68_face_landmarks.dat")

account_sid = 'ACef5d7bbdb28013a19c88f7874d4f1ada'
auth_token = '5deca3ead6cd4d399aa281cbcb47040a'
client = Client(account_sid, auth_token)

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	

def get_landmarks(im):
    rects = Detection(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in Prediction(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance


thresh = 0.25
frame_check = 20
yawns = 0
eyes=0
present_yawn = False 

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(1)
flag=0

while True:
    ret, frame = cap.read()   
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    image_landmarks, lip_distance = mouth_open(frame)
    previous_yawn = present_yawn 

    subjects = Detection(gray, 0)
    for (i,subject) in enumerate(subjects):
        shape = Prediction(gray, subject)
        shape = face_utils.shape_to_np(shape)#converting to NumPy Array
        #face detection
        (X_length,Y_length,Width,Height)=face_utils.rect_to_bb(subject)
        cv2.rectangle(frame,(X_length,Y_length),(X_length + Width, Y_length + Height), (0,255,0), 2)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        lefteyehull = cv2.convexHull(leftEye)
        righteyehull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [lefteyehull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [righteyehull], -1, (0, 255, 0), 1)
        if ear < thresh:
            flag += 1
            print (flag)
            if flag >= frame_check:
                eyes+=1
                cv2.putText(frame, "Drowsiness detected", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#print ("Drowsy")
        else:
            flag = 0
        if lip_distance > 40:
            present_yawn = True 
        
            cv2.putText(frame, "Driver is Yawning", (50,250), 
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        

            output_text = " Yawn Count: " + str(yawns + 1)

            cv2.putText(frame, output_text, (50,50),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
        
        else:
            present_yawn = False 

        if previous_yawn == True and present_yawn == False:
            yawns += 1

        if yawns>=5 or eyes>=5:
            for i in range(1):
                playsound('data_alarm.mp3')
                message = client.messages \
                    .create(
                        body="Stop the vehicle with number: 8114",
                        from_='+12315254666',
                        to='+918919859493'
                    )
            
            yawns=0
            eyes=0
         
	
    cv2.imshow("Frame", frame)
    
    
    if cv2.waitKey(1) == 1: #1 is the Enter Key
        break
        

cv2.destroyAllWindows()
cap.release() 

