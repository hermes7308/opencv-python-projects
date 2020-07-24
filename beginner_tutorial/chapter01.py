# Chapter 1- Read Images Video Webcam
import cv2
import numpy as np
print("Package Imported")

# 1
# img = cv2.imread("Resources/lena.png")
#
# cv2.imshow("Ouput", img)
# cv2.waitKey(0)

# 2
# cap = cv2.VideoCapture("../Resources/test_video.mp4") # read a video file
cap = cv2.VideoCapture(0)  # read a webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)

while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

