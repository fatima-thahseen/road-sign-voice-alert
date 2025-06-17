# 🚦 Road Sign Detection and Voice Alert System

A real-time computer vision system that detects road signs through webcam, classifies them using a fine-tuned VGG16 model, and gives voice alerts — just like a smart driver-assist system!

---

## Project Overview

This deep learning project identifies traffic signs using a webcam feed and notifies the user via voice alert using text-to-speech. It mimics a driver assistant by observing signs in real time and reacting only when predictions are stable — just like a patient human.

---

## Features

- Real-time webcam-based road sign detection using OpenCV  
- VGG16-based traffic sign classifier (Transfer Learning)  
- Voice alerts using pyttsx3  
- Buffer logic: speaks only if 3 out of last 5 frames agree  
- FPS overlay for performance monitoring  

---

## Tech Stack

| Category        | Tools / Libraries                   |
|----------------|--------------------------------------|
| Programming    | Python                               |
| Deep Learning  | TensorFlow, Keras (VGG16)            |
| Computer Vision| OpenCV                               |
| Voice Output   | pyttsx3 (text-to-speech)             |
| Others         | NumPy, Pickle, Google Drive (model)  |

---

## Download the Trained Model

Due to GitHub's 100MB file size limit, the model is hosted externally.

👉 **[Download vgg16_best_model.keras](https://drive.google.com/file/d/1j3qWtB9Om3A55sSuZgRtJjGfUBTFv2-C/view?usp=sharing)**  
_(Hosted on Google Drive)_

---

## Model Info

- **Backbone**: VGG16 pretrained on ImageNet  
- **Dataset**: GTSRB (German Traffic Sign Recognition Benchmark)  
- **Training**: Used `ImageDataGenerator` with augmentation  
- **Architecture**: Fine-tuned with additional dense + dropout layers  
- **Accuracy**: High validation performance and stable real-time results  

---

## 📦 Dataset Preparation

The script [`dataset_preparation.py`](dataset_preparation.py) is used to:
- Crop signs using ROI data from the GTSRB dataset
- Resize images to 160×160
- Split the dataset into training and validation sets

  
⭐ If you liked this project, give it a star on GitHub!


