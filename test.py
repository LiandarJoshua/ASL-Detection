import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from tensorflow.keras.models import load_model
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the model
model_path = r"C:\Users\joshu\HandSignDetection\Model\keras_model.h5"
labels_path = r"C:\Users\joshu\HandSignDetection\Model\labels.txt"
model = load_model(model_path)

# Load the labels
with open(labels_path, 'r') as f:
    labels = [line.split()[1] for line in f.read().splitlines()]

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the hand detector
detector = HandDetector(maxHands=1)

# Set parameters
offset = 20
imgSize = 300
folder = "Data/U"
counter = 0
sentence = ""
predicted_letter = ""  # To store the current predicted letter

while True:
    # Read the image from the webcam
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
    
    # Detect hands in the image
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Create a white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Crop the hand region with offset
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        
        if imgCrop.size > 0:
            aspectRatio = h / w
            
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hCal + hGap, :] = imgResize
            
            # Prepare the image for prediction
            imgWhiteResized = cv2.resize(imgWhite, (224, 224))
            imgWhiteResized = imgWhiteResized / 255.0  # Normalize to [0, 1]
            imgWhiteResized = np.expand_dims(imgWhiteResized, axis=0)
            
            # Make predictions
            predictions = model.predict(imgWhiteResized)
            predicted_letter = labels[np.argmax(predictions)]  # Get the predicted letter
            confidence = np.max(predictions)
            
            # Display predictions
            cv2.putText(img, f'{predicted_letter} ({confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Display the cropped and processed images
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)
    
    # Display the original image with the sentence
    cv2.rectangle(img, (10, img.shape[0] - 60), (img.shape[1] - 10, img.shape[0] - 10), (255, 255, 255), -1)
    cv2.putText(img, sentence, (20, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Display the original image
    cv2.imshow('Image', img)
    
    # Check for key press
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        # Save the processed image
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved image {counter}")
    elif key == ord("1"):
        if predicted_letter == "STOP":
            engine.say(sentence)
            engine.runAndWait()
        else:
            sentence += predicted_letter  # Add only the predicted letter to the sentence
    elif key == ord("2"):
        sentence = ""  # Clear the sentence
    elif key == 32:  # Space key
        sentence += " "  # Add a space to the sentence

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()