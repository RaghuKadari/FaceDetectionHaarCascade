###############################################################################
# License: GPL. 
# 
#
#
# Brief: Face detection using Haarcascade
#
#
###############################################################################


import sys
import cv2    
import numpy as np
import time
import matplotlib.pyplot as plt

def FindFace():

    face_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_eye.xml')

    #you can use your video feed here or you 
    #capture from your webcam directly
    #video = cv2.VideoCapture(0)  #Capture from Webcam
    video = cv2.VideoCapture('IM.mp4') #video Feed

    
    # We need to check if camera 
    # is opened previously or not 
    if (video.isOpened() == False):  
        print("Error reading video file") 
      
    # We need to set resolutions. 
    # so, convert them from float to integer. 
    frame_width = int(video.get(3)) 
    frame_height = int(video.get(4)) 
       
    size = (frame_width, frame_height) 
    size = (1280, 720) 
       
    # Below VideoWriter object will create 
    # a frame of above defined The output  
    # is stored in 'filename.avi' file. 
    vw = cv2.VideoWriter('IM_FF.avi',  
                             cv2.VideoWriter_fourcc(*'MJPG'), 
                             10, size)



    while(True):
        # Capture frame-by-frame
        ret, img = video.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv2.imshow('video',img)
     
        vw.write(img)

        if ((cv2.waitKey(1) & 0xFF) == ord('q')):
            break

    # When everything done, release the capture
    video.release()
    cv2.destroyAllWindows()



def main():
    
    FindFace()


if __name__ == '__main__':
    main()


