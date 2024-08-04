ASL-Detection
This project aims to create a real-time American Sign Language (ASL) alphabet detection system using a Convolutional Neural Network (CNN) and deploy it as a web application using Flask. The model is trained on a custom dataset of ASL alphabets.

Overview
The ASL Sign Language Detection project involves:

Creating a custom dataset of ASL alphabet images.
Training a CNN to recognize ASL alphabets.
Deploying the trained model using Flask to provide a web interface for real-time sign language detection.
Dataset
The dataset consists of images of hands showing the ASL alphabet. Each image corresponds to a letter from A to Z. The dataset is organized into subfolders for each alphabet letter, containing images of that particular sign. Due to its large size, the dataset could not be attached here. However, users can create their own dataset using datacollection.py, which reads pictures on pressing 1 and saves them to the designated folder.

Model
The model is a Convolutional Neural Network (CNN) designed to classify images of ASL alphabets. The architecture includes multiple convolutional layers followed by pooling layers and fully connected layers. The model is trained to minimize categorical cross-entropy loss and is evaluated on a validation set.

Installation
Install the required packages listed in requirements.txt and run the application locally using the app.py file.

Text To Speech
This project uses text-to-speech to read out the letters forming a sentence.

Press 1 to add the letter to the text box.
Press 2 to clear the text box.
Press space to add a space.
To hear the sentence, show the stop symbol by showing your full palm with fingers fully extended and press 1.
