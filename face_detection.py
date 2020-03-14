#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:42:57 2020

@author: jessefilho
https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
"""
# conda install opencv

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time




# %%


#get only webcam
vs = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('/Users/jessefilho/PycharmProjects/reconnaissance_formes_images/venv/lib/python2.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/jessefilho/PycharmProjects/reconnaissance_formes_images/venv/lib/python2.7/site-packages/cv2/data/haarcascade_eye.xml')




#%%
# keep looping
img_counter = 0
while True:
	# grab the current frame
    ret, frame = vs.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5, #defines how many objects are detected near the current one before it declares the face found
        minSize=(30, 30), #gives the size of each window.
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    #It put rectangles in which it believes it found a face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    



                
	# handle the frame from VideoCapture or VideoStream
	#frame = frame[1] if args.get("video", False) else frame
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
    if frame is None:
        vs.stop()
        cv2.destroyAllWindows()
        break
	# resize the frame, blur it, and convert it to the HSV
	# color space
    frame = imutils.resize(frame, width=1200)
	
    
	# show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
	
    
    #ESC Pressed
    if key == 27: 
        break  
    #SPACE pressed
    elif key == 32:       
        img_name = "facedetect_webcam_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()
# otherwise, release the camera
else:
	vs.release()
# close all windows
cv2.destroyAllWindows()
    







