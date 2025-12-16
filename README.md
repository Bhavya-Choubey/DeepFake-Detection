# Deepfake Detection System

A web-based deepfake detection system that analyzes videos and images to identify whether they are real or manipulated. 
The system leverages a **CNN-LSTM model** for accurate temporal and spatial feature extraction and provides a simple **Flask web interface** for users.

## Overview
Deepfakes are AI-generated synthetic media that replace or manipulate facial content in videos or images.
This project detects such deepfakes by analyzing patterns in facial movements and inconsistencies using a 
combination of **Convolutional Neural Networks (CNNs)** for spatial features and **Long Short-Term Memory (LSTM)** networks for temporal sequence analysis.

The system is deployed using **Flask**, providing a user-friendly web interface to upload videos or images and get predictions in real-time.

## Features
- Detect deepfake videos and images with high accuracy.
- Web interface to upload media files.
- Real-time predictions with probability scores.
- Model trained on publicly available deepfake datasets.
- Scalable architecture for future improvements.

- ## Tech Stack
- **Backend:** Python, Flask
- **Modeling:** TensorFlow / PyTorch, CNN-LSTM
- **Frontend:** HTML, CSS, JavaScript
