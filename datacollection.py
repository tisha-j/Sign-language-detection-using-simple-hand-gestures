import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize the video capture object
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Initialize the hand detector
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

# Define the folder path for saving images
folder = r"C:\Users\sanaa\OneDrive\Desktop\flof\Data\yes"

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the image around the hand
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

        # Ensure imgCrop is not empty
        if imgCrop.size == 0:
            print("Warning: Empty image crop detected")
            continue

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Display the cropped and white images
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    # Display the original image
    cv2.imshow('Image', img)

    # Save the image if 's' is pressed
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Image saved: {counter}")

    # Break the loop if 'q' is pressed
    if key == ord("q"):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
