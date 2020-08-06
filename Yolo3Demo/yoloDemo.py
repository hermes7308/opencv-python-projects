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

whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = "coco.names"
classesNames = []

with open(classesFile, "rt") as f:
    classesNames = f.read().rstrip("\n").split("\n")
print(classesNames)
print(len(classesNames))

# download yolo files in https://pjreddie.com/darknet/yolo/
modelConfiguration = "yolov3.cfg"  # fast speed: yolov3-tiny.cfg, 320: yolov3.cfg
modelWeights = "yolov3.weights"  # fast speed: yolov3-tiny.weights, 320: yolov3.weights

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nms_threshold=nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f"{classesNames[classIds[i]].upper()} {int(confs[i] * 100)}%",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    # print(layerNames)
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)

    outputs = net.forward(outputNames)
    # print(type(outputs))

    findObjects(outputs, img)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
