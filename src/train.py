import numpy as np
import joblib
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from features import extract_lbp, extract_hog
from preprocess import detect_and_preprocess
from utils import load_fer_dataset
from models import mini_xception
import tensorflow as tf

# -------------------------------
# Load dataset
# -------------------------------
dataset_path = "data/fer2013/train"
image_paths, labels = load_fer_dataset(dataset_path)
print(f"[INFO] Loaded {len(image_paths)} images with labels.")

# -------------------------------
# Prepare classical features with progress bar
# -------------------------------
def prepare_features_and_labels(image_paths, labels):
    X_lbp, X_hog, y = [], [], []
    print("[INFO] Extracting classical features (LBP & HOG)...")
    for p, l in tqdm(zip(image_paths, labels), total=len(image_paths), desc="Feature extraction"):
        img = cv2.imread(p)
        face48, face_classical = detect_and_preprocess(img)
        if face48 is None:
            continue
        X_lbp.append(extract_lbp(face_classical))
        X_hog.append(extract_hog(face_classical))
        y.append(l)
    return np.vstack(X_lbp), np.vstack(X_hog), np.array(y)

X_lbp, X_hog, y = prepare_features_and_labels(image_paths, labels)

# -------------------------------
# Train LBP + KNN
# -------------------------------
Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_lbp, y, test_size=0.15, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
knn.fit(Xl_train, yl_train)
print('KNN train acc:', knn.score(Xl_train, yl_train), 'test acc:', knn.score(Xl_test, yl_test))
joblib.dump(knn, './models/lbp_knn.pkl')

# -------------------------------
# Train HOG + Linear SVM
# -------------------------------
Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_hog, y, test_size=0.15, stratify=y)
svm = LinearSVC(C=1.0, max_iter=20000, verbose=1)
svm.fit(Xh_train, yh_train)
print('SVM train acc:', svm.score(Xh_train, yh_train), 'test acc:', svm.score(Xh_test, yh_test))
joblib.dump(svm, './models/hog_svm.pkl')

# -------------------------------
# Prepare CNN data with progress bar
# -------------------------------
print("[INFO] Preparing CNN data (mini-Xception)...")
X_cnn = []
for p in tqdm(image_paths, desc="CNN preprocessing"):
    img = cv2.imread(p)
    face48, _ = detect_and_preprocess(img)
    if face48 is not None:
        X_cnn.append(face48)
X_cnn = np.array(X_cnn, dtype='float32') / 255.0
X_cnn = X_cnn.reshape(-1, 48,48,1)
y_cnn = y[:len(X_cnn)]  # ensure label alignment

# One-hot encode labels
y_cnn_onehot = tf.keras.utils.to_categorical(y_cnn, num_classes=7)

# Train/test split for CNN
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cnn, y_cnn_onehot, test_size=0.15, stratify=y_cnn)

# -------------------------------
# Train mini-Xception CNN
# -------------------------------
cnn = mini_xception(input_shape=(48,48,1), num_classes=7)
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("[INFO] Training mini-Xception CNN...")
cnn.fit(
    Xc_train, yc_train,
    validation_data=(Xc_test, yc_test),
    batch_size=64,
    epochs=30,
    verbose=1
)

# Save CNN weights
cnn.save_weights('./models/minixception.weights.h5')
print("[INFO] mini-Xception weights saved to ./models/minixception.weights.h5")
