import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize the video capture object
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Initialize the hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier(r"C:\Users\sanaa\OneDrive\Desktop\flof\converted_keras\keras_model.h5", r"C:\Users\sanaa\OneDrive\Desktop\flof\converted_keras\labels.txt")
offset = 20
imgSize = 300
counter = 0

labels = ["hello", "thumbsdown", "thumbsup", "yes"]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Print the bounding box coordinates
        print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")

        # Ensure the coordinates are within the bounds of the image
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        # Create a white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the image around the hand
        imgCrop = img[y1:y2, x1:x2]

        # Ensure imgCrop is not empty
        if imgCrop.size == 0:
            print("Warning: Empty image crop detected")
            continue

        # Print the shape of the cropped image
        print(f"Cropped image shape: {imgCrop.shape}")

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    key = cv2.waitKey(1)

    # Break the loop if 'q' is pressed
    if key == ord("q"):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
