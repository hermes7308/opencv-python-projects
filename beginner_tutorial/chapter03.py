# Chapter 3 – Crop and Resize
import cv2
import numpy as np

img = cv2.imread("../Resources/lambo.png")
print(img.shape) # (462, 623, 3) height, width, BGR

imgResize = cv2.resize(img, (300, 200)) # width, height
print(imgResize.shape) # (200, 300, 3) height, width, BGR

imgCropped = img[0:200, 200:500] # width, height

cv2.imshow("Image", img)
cv2.imshow("Image Resize", imgResize)
cv2.imshow("Image Cropped", imgCropped)

cv2.waitKey(0)