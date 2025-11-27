# Facial Emotion Detector

This project was developed for a Robotics course in my Master's program in Virtual and Augmented Reality (VAR).
The goal is to compare three different facial emotion recognition models side by side using an interactive interface that supports both images and webcam input.

The system measures prediction, confidence, and inference latency for all models.

# Models Included

1. LBP + KNN
    
    A traditional computer vision baseline using Local Binary Patterns.

2. HOG + Linear SVM (OvR)
    
    Uses scikit-image HOG features and scikit-learn's LinearSVC classifier.

3. mini-Xception (CNN)

    A lightweight deep learning model trained on FER-2013 (typical accuracy ~60–67%).

# Processing Pipeline

For each frame (image or webcam):
1. Detect the face (OpenCV Haar Cascade)
2. Crop and optionally align
3. Resize (48x48 grayscale for mini-Xception)
4. Run all three models
5. Display:
    - Predicted emotion
    - Confidence score
    - Inference time (ms)

# User Interface

Built using Gradio, the interface supports:
- Uploading an image
- Capturing frames from the webcam
- Comparing the three model outputs side-by-side

# How to Run
1. Install dependencies
    ```nginx
    pip install -r requirements.txt
    ```
2. Launch the Gradio comparison interface
    ```nginx
    python -m apps.app_gradio
    ```
3. Run the real-time webcam interface
    ```nginx
    python -m apps.realtime
    ```