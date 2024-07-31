from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import math
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import pyttsx3
import os

# Disable the auto-load of .env files
os.environ['FLASK_SKIP_DOTENV'] = '1'

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the model
model_path = r"C:\Users\joshu\HandSignDetection\Model\keras_model.h5"
labels_path = r"C:\Users\joshu\HandSignDetection\Model\labels.txt"
model = load_model(model_path)

# Load the labels
with open(labels_path, 'r') as f:
    labels = [line.split()[1] for line in f.read().splitlines()]

# Initialize the hand detector
detector = HandDetector(maxHands=1)

# Set parameters
offset = 20
imgSize = 300
folder = "Data/U"
counter = 0
sentence = ""
predicted_letter = ""  # To store the current predicted letter

# Initialize the webcam
cap = None

# Flag to control continuous TTS output
speak_sentence = False

app = Flask(__name__)

def generate_frames():
    global cap, sentence, predicted_letter, counter, speak_sentence
    if cap is None:
        cap = cv2.VideoCapture(0)
    
    while True:
        # Read the image from the webcam
        success, img = cap.read()
        if not success:
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
        
        # Display the original image with the sentence
        cv2.rectangle(img, (10, img.shape[0] - 60), (img.shape[1] - 10, img.shape[0] - 10), (255, 255, 255), -1)
        cv2.putText(img, sentence, (20, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Handle continuous TTS output
        if speak_sentence and predicted_letter == "STOP" and sentence.strip() != "":
            engine.say(sentence)
            engine.runAndWait()
            speak_sentence = False  # Reset the flag after speaking
            sentence = ""  # Reset the sentence after speaking
        
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        
        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/keypress/<key>', methods=['POST'])
def keypress(key):
    global sentence, predicted_letter, speak_sentence
    
    if key == '1':
        if predicted_letter == 'STOP':
            speak_sentence = True  # Set flag to speak sentence
        else:
            sentence += predicted_letter  # Add only the predicted letter to the sentence
    elif key == '2':
        sentence = ''  # Clear the sentence
    elif key == 'space':
        sentence += ' '  # Add a space to the sentence
    
    return '', 204  # HTTP 204 No Content

@app.route('/shutdown', methods=['GET'])
def shutdown():
    global cap
    if cap is not None:
        cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows
    return 'Server shutting down...'

if __name__ == "__main__":
    app.run(debug=True)
