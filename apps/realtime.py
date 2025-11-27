import sys
import os
import cv2

# Add the src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from preprocess import detect_and_preprocess
from inference import predict_lbp, predict_hog, predict_cnn

emotion_names = ['angry','disgust','fear','happy','sad','surprise','neutral']

def label_text(lbl):
    try:
        return emotion_names[int(lbl)]
    except:
        return str(lbl)

# open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face48, face_classical = detect_and_preprocess(frame)
    
    if face48 is not None:
        lbp_label, lbp_conf, _ = predict_lbp(face_classical)
        hog_label, hog_conf, _ = predict_hog(face_classical)
        cnn_label, cnn_conf, _ = predict_cnn(face48)

        text_lbp = f"LBP: {label_text(lbp_label)} ({lbp_conf:.2f})"
        text_hog = f"HOG: {label_text(hog_label)} ({hog_conf:.2f})"
        text_cnn = f"CNN: {label_text(cnn_label)} ({cnn_conf:.2f})"

        # draw results
        cv2.putText(frame, text_lbp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, text_hog, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, text_cnn, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Emotion Recognition", frame)
    
    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
