import cv2
from tracker import *
import time

# Create tracker object
tracker = EuclideanDistTracker()
fileName = "00"
cap = cv2.VideoCapture("./video/video_"+str(fileName)+".mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=40)
i = 0
totalframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
prev_x = 0
prev_y = 0
imageName = 1
move = ""
framesToCapture = math.floor(totalframes / 5)
print("total" + str(framesToCapture))
while True:
    ret, frame = cap.read()

    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame[40: 600, 100:600]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 15:
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)

            detections.append([x, y, w, h])
            # time.sleep(5)

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        move = ""
        x, y, w, h, id = box_id
        if prev_x < x and prev_y < y:
            move = "right down"
        elif prev_x > x and prev_y < y:
            move = "left down"
        elif prev_x < x and prev_y > y:
            move = "right up"
        elif prev_x > x and prev_y > y:
            move = "left up"
        elif prev_x == x and prev_y < y:
            move = "down"
        elif prev_x == x and prev_y > y:
            move = "up"
        elif prev_x < x and prev_y == y:
            move = "right"
        elif prev_x > x and prev_y == y:
            move = "left"
        else:
            move = ""

        cv2.imshow("Frame", frame)
    # time.sleep(5)
    print(move+str(i))
    if (i == framesToCapture and move != ""):
        cv2.putText(roi, str(move), (x, y - 15),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print("move" + str(move))
        move = ""
        name = './dataset/' + str(fileName) + "-img-0" + \
            str(imageName) + '.jpg'
        # print('Creating...' + name)
        cv2.imwrite(name, frame)
        prev_x = x
        prev_y = y
        i = 0
        imageName += 1
    i += 1

    cv2.imshow("roi", roi)

    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
