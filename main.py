'''This script detects if a person is drowsy or not,using dlib and eye aspect ratio
calculations. Uses webcam video feed as input.'''

#importing necessary libraries

import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import imutils
import numpy as np
import pygame #for playing sound
import time




# initialize pygame for alerting sound music
pygame.mixer.init()
pygame.mixer.music.load('alarm.wav')



# this function calculate eye aspect ratio

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    C = distance.euclidean(eye[0],eye[3])

    # compute the eye aspect ratio
    ear = (A+B) / (2.0*C)

    return ear
 # calculating mouth aspect ratio
def mouth_aspect_ratio(mou):
    # compute the euclidean distances between the horizontal
    X =distance.euclidean(mou[0],mou[6])
    # compute the euclidean distances between the vertical
    Y1 = distance.euclidean(mou[2], mou[10])
    Y2 = distance.euclidean(mou[4], mou[8])
    # taking average
    Y = (Y1 + Y2) / 2.0
    # compute mouth aspect ratio
    mar = Y / X
    return mar




# starting the web camera for video capture

cap = cv2.VideoCapture(0)

# define constants for aspect ratios
eye_aspect_ratio_threshold = 0.30
eye_aspect_ratio_consec_frames = 35
MOU_AR_THRESH = 0.75

counter = 0
yawnStatus = False
yawns = 0

# Load face cascade which will be used to draw a rectangle around detected face

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



# load face detector and predictor, use dliib shape predictor file i.e .dat extension
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# grab the indexes of the facial landmarks for the left and right eye
# also for the mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# loop over captuing video
while True:

    # reading each frame and convert it in to gray scale

    ret, frame = cap.read()
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame,cv2.COLOR_RGBA2GRAY)
    prev_yawn_status = yawnStatus
    faces = detector(gray ,0) # detect the facial point by detector function

    # detect the face through haarcascade_frontalface_defult.xml
    face_rectangle = face_cascade.detectMultiScale(gray,1.3,5)

    # draw rectangle around each face detected

    # for haar cascade rectangle
    for (x, y, w, h) in face_rectangle:
        x1, y1 = x + w, y + h
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # for rectangle look better extra
        cv2.line(frame, (x, y), (x + 30, y), (255, 255, 0), 6)  # top left
        cv2.line(frame, (x, y), (x, y + 30), (255, 255, 0), 6)

        cv2.line(frame, (x1, y), (x1 - 30, y), (255, 255, 0), 6)  # top right
        cv2.line(frame, (x1, y), (x1, y + 30), (255, 255, 0), 6)

        cv2.line(frame, (x, y1), (x + 30, y1), (255, 255, 0), 6)  # bottom left
        cv2.line(frame, (x, y1), (x, y1 - 30), (255, 255, 0), 6)

        cv2.line(frame, (x1, y1), (x1 - 30, y1), (255, 255, 0), 6)  # bottom right
        cv2.line(frame, (x1, y1), (x1, y1 - 30), (255, 255, 0), 6)
    # determine the facial landmark point ,
    # convert the facial landmark (x,y) -coordinates to NumPy array




    for face in faces:

        landmarks = predictor(gray, face)

        # draw landmarks on face
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 255, 0), 2)

        shape = predictor(gray,face)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        # after that by the coordinators compute the eye aspect ratio
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        mouEAR = mouth_aspect_ratio(mouth)
        # avg the EAR together for both the eyes
        ear = (leftEyeAspectRatio + rightEyeAspectRatio) / 2


        # compute the convex hull for left and right eye,
        # then visualize the eye
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [leftEyeHull],-1,(0,255,255),1)
        cv2.drawContours(frame, [rightEyeHull],-1,(0,255,255),1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        # detect if eye aspect ratio is less than threshold
        if ear < eye_aspect_ratio_threshold:
            counter += 1
            cv2.putText(frame, "Eyes Closed ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if counter >= eye_aspect_ratio_consec_frames:
                pygame.mixer.music.play(-1)
                cv2.putText(frame,"*****DROWSINESS ALERT!****",(170,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
                cv2.putText(frame, "---WAKE UP---", (200, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255),2)


        else:
            counter = 0
            pygame.mixer.music.stop()

            cv2.putText(frame, "Eyes Open ", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "8th sem B.Tech Major project", (370, 470), cv2.FONT_HERSHEY_COMPLEX, 0.6, (153, 51, 102), 1)

        # yawning detections extra
        if mouEAR > MOU_AR_THRESH:
            cv2.putText(frame, "Yawning", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            yawnStatus = True
            output_text = "Yawn Count: " + str(yawns + 1)
            cv2.putText(frame,output_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            yawnStatus = False

        if prev_yawn_status == True and yawnStatus == False:
            yawns += 1

        cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (480, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame,"Drowsiness Detection" ,(30, 470), cv2.FONT_HERSHEY_COMPLEX, 0.6, (153, 51, 102), 1)


    cv2.imshow("Frames",frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
