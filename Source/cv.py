import cv2
import sys
import numpy as np

# I use this for testing background subtractors, etc.
# This is not for use in the project
# Use the relevant file for relevant code
# preprocessing.py for realtime raspberry pi processing (MUST BE FAST)
# postprocessing.py (tbd) for postprocessing, etc.

#cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(sys.argv[1])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))

total_pixels = 0

#remove first 90 frames
i = 0
 
while(i < 90):
    ret, frame = cap.read()
    if not total_pixels:
        total_pixels = frame.size / 3
    i+= 1

mog2 = cv2.createBackgroundSubtractorMOG2()
mog = cv2.bgsegm.createBackgroundSubtractorMOG()
cnt = cv2.bgsegm.createBackgroundSubtractorCNT()
gmg = cv2.bgsegm.createBackgroundSubtractorGMG()
gsoc = cv2.bgsegm.createBackgroundSubtractorGSOC()

while(True):
    ret, frame = cap.read()

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    #frame = cv2.blur(frame, (10,10))

    frame = cv2.resize(frame, (640, 480))
    
    frame_mog2 = mog2.apply(frame)
    frame_mog = mog.apply(frame)
    frame_cnt = cnt.apply(frame)
    frame_gmg = gmg.apply(frame)
    frame_gosc = gsoc.apply(frame)

    cv2.imshow('frame_mog2', frame_mog2)
    cv2.imshow('frame_mog', frame_mog)
    cv2.imshow('frame_cnt', frame_cnt)
    cv2.imshow('frame_gmg', frame_gmg)
    cv2.imshow('frame_gosc', frame_gosc)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()