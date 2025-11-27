import sys
import os
import gradio as gr
import cv2
import numpy as np

# Add the src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from preprocess import detect_and_preprocess
from inference import predict_lbp, predict_hog, predict_cnn

emotion_names = ['angry','disgust','fear','happy','sad','surprise','neutral']

def run_on_frame(frame):  # frame is numpy BGR
    face48, face_classical = detect_and_preprocess(frame)
    if face48 is None:
        return ("No face",0,0), ("No face",0,0), ("No face",0,0), frame
    # predictions
    lbp_label, lbp_conf, lbp_ms = predict_lbp(face_classical)
    hog_label, hog_conf, hog_ms = predict_hog(face_classical)
    cnn_label, cnn_conf, cnn_ms = predict_cnn(face48)

    def label_text(lbl):
        try:
            return emotion_names[int(lbl)]
        except:
            return str(lbl)

    out_lbp = f"{label_text(lbp_label)} ({lbp_conf:.2f}) — {lbp_ms:.1f} ms"
    out_hog = f"{label_text(hog_label)} ({hog_conf:.2f}) — {hog_ms:.1f} ms"
    out_cnn = f"{label_text(cnn_label)} ({cnn_conf:.2f}) — {cnn_ms:.1f} ms"

    face_bgr = cv2.cvtColor(face48, cv2.COLOR_GRAY2BGR)
    return out_lbp, out_hog, out_cnn, face_bgr

with gr.Blocks() as demo:
    gr.Markdown("## Emotion prediction: LBP+KNN | HOG+LinearSVC | mini-Xception")
    with gr.Row():
        inp = gr.Image(type="numpy")  # upload only
        with gr.Column():
            lbp_text = gr.Textbox(label="LBP + KNN", interactive=False)
            hog_text = gr.Textbox(label="HOG + LinearSVC", interactive=False)
            cnn_text = gr.Textbox(label="mini-Xception", interactive=False)
            face_preview = gr.Image(label="Detected face", type="numpy")
    btn = gr.Button("Run (single frame)")
    btn.click(run_on_frame, inputs=inp, outputs=[lbp_text, hog_text, cnn_text, face_preview])

demo.launch(server_name="0.0.0.0", share=False)
