# ASL-Detection
This project aims to create a real-time American Sign Language (ASL) alphabet detection system using a Convolutional Neural Network (CNN) and deploy it as a web application using Flask. The model is trained on a custom dataset of ASL alphabets.

Overview
The ASL Sign Language Detection project involves:

Creating a custom dataset of ASL alphabet images.
Training a CNN to recognize ASL alphabets.
Deploying the trained model using Flask to provide a web interface for real-time sign language detection.



Dataset
The dataset consists of images of hands showing the ASL alphabet. Each image corresponds to a letter from A to Z. The dataset is organized into subfolders for each alphabet letter, containing images of that particular sign.


Model
The model is a Convolutional Neural Network (CNN) designed to classify images of ASL alphabets. The architecture includes multiple convolutional layers followed by pooling layers and fully connected layers. The model is trained to minimize categorical cross-entropy loss and is evaluated on a validation set.



Install the required packages in requirements.txt and run it locally using the app.py file.


Text To Speech:
This project uses text to speech to read out the letters so far forming a sentence.
Press 1 to add the letter to the text box
Press 2 to clear
Press space to add a space
If you want to hear the sentence, show the stop symbol by showing your full palm and fingers fully extended and press 1.

