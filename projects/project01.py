# Project 1 – Virtual Paint
import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(cv2.CAP_DSHOW)  # 카메라 켜지는데 너무 느려서 설정 (Logitech BRIO Webcam)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

myColors = [[52, 82, 41, 179, 192, 176]]  # blue

myColorValues = [[255, 51, 153]]  # purple

myPoints = []  # [x, y, colorId]


# TODO: HSV 에 대해서 공부가 필요
def findColor(img, myColors, myColorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lowerb=lower, upperb=upper)
        x, y = getContours(mask)
        cv2.circle(imgResult, (x, y), 10, myColorValues[count], cv2.FILLED)
        if x != 0 and y != 0:
            newPoints.append([x, y, count])
        count += 1
        # cv2.imshow(str(color[0]), mask)
    return newPoints


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 윤곽을 가져옴
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)  # 윤곽의 넓이
        if area > 500:
            # cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)  # 윤곽 그리기 (파랑)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # arcLength 에 일정한 값을 곱하여 근사값을 적용한다.
            x, y, w, h = cv2.boundingRect(approx)  # Calculates the up-right bounding rectangle of a point set.
    return x + w // 2, y


def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)


while True:
    success, img = cap.read()
    imgResult = img.copy()
    newPoints = findColor(img, myColors, myColorValues)
    if len(newPoints) != 0:
        for newPoint in newPoints:
            myPoints.append(newPoint)
    if len(myPoints) != 0:
        drawOnCanvas(myPoints, myColorValues)

    cv2.imshow("Video", imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
