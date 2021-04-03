import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
inputImg = cv2.imread("Resources/chmo.jpg")
#img = cv2.resize(inputImg,(400,700))
imgGray = cv2.cvtColor(inputImg,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray,1.1,4)
for (x,y,h,w) in faces:
    cv2.rectangle(inputImg,(x,y),(x+w,y+h),(255,0,0),2)


cv2.imshow("Face",inputImg)
cv2.imshow("GrayFace",imgGray)
cv2.waitKey(0)