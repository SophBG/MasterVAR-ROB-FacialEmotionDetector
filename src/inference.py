import time
import numpy as np
import joblib
import tensorflow as tf
from models import mini_xception
from features import extract_lbp, extract_hog

# load traditional models
knn = joblib.load('./models/lbp_knn.pkl')
svm = joblib.load('./models/hog_svm.pkl')
# optionally load scalers if used
svm_scaler = None  # load if you used

# load CNN
num_classes = 7
cnn = mini_xception(input_shape=(48,48,1), num_classes=num_classes)
cnn.load_weights('./models/minixception.weights.h5')

def predict_lbp(face_classical):
    # face_classical: 64x64 grayscale np.uint8
    t0 = time.perf_counter()
    feat = extract_lbp(face_classical).reshape(1, -1)
    probas = knn.predict_proba(feat)[0]  # requires KNN with predict_proba
    label = knn.classes_[np.argmax(probas)]
    ms = (time.perf_counter() - t0) * 1000
    return label, float(np.max(probas)), ms

def predict_hog(face_classical):
    t0 = time.perf_counter()
    feat = extract_hog(face_classical).reshape(1, -1)
    if svm_scaler:
        feat = svm_scaler.transform(feat)
    # LinearSVC doesn't have predict_proba by default; use decision_function and softmax-like mapping
    if hasattr(svm, "predict_proba"):
        probas = svm.predict_proba(feat)[0]
    else:
        decisions = svm.decision_function(feat)[0]
        # convert to probabilities with softmax
        exps = np.exp(decisions - np.max(decisions))
        probas = exps / exps.sum()
    label = svm.classes_[np.argmax(probas)]
    ms = (time.perf_counter() - t0) * 1000
    return label, float(np.max(probas)), ms

def predict_cnn(face_48):
    t0 = time.perf_counter()
    x = face_48.astype('float32') / 255.0
    x = x.reshape(1,48,48,1)
    probas = cnn.predict(x, verbose=0)[0]
    label = np.argmax(probas)
    ms = (time.perf_counter() - t0) * 1000
    return int(label), float(np.max(probas)), ms
