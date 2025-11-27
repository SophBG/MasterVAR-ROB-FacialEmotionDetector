import cv2
import numpy as np

# -------------------------------
# Initialize Haar Cascade for face detection
# -------------------------------
# Using OpenCV default frontal face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------
# Detect face bounding boxes in a grayscale image
# -------------------------------
def detect_face_bboxes(gray):
    """
    Detect faces in a grayscale image using Haar Cascade.

    Parameters:
        gray (np.array): Grayscale image

    Returns:
        faces (list of tuples): List of (x, y, w, h) bounding boxes
    """
    # Adjusted parameters to detect smaller faces and reduce misses
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,   # smaller step between scales
        minNeighbors=3,      # more permissive
        minSize=(24, 24)     # allow smaller faces
    )
    return faces

# -------------------------------
# Crop and optionally align face
# -------------------------------
def crop_align_face(img, bbox, desired_size=(48, 48), do_align=False, landmarks_predictor=None):
    """
    Crop the detected face with padding and resize to desired size.
    Optional alignment can be added using landmarks.

    Parameters:
        img (np.array): Original BGR image
        bbox (tuple): Face bounding box (x, y, w, h)
        desired_size (tuple): Output size (width, height)
        do_align (bool): Whether to align face using landmarks
        landmarks_predictor: Optional landmarks predictor

    Returns:
        face_resized (np.array): Cropped and resized face
    """
    x, y, w, h = bbox
    pad = int(0.2 * w)  # 20% padding around face
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad)
    y1 = min(img.shape[0], y + h + pad)

    face = img[y0:y1, x0:x1].copy()

    if do_align and landmarks_predictor:
        # Optional: use landmarks_predictor to align eyes horizontally
        pass

    # Resize to desired size
    face_resized = cv2.resize(face, desired_size, interpolation=cv2.INTER_LINEAR)
    return face_resized

# -------------------------------
# Detect and preprocess face for classical and deep learning features
# -------------------------------
def detect_and_preprocess(frame_bgr):
    """
    For FER-2013, just resize the image (already 48x48 grayscale)
    """
    # Convert to grayscale if not already
    if len(frame_bgr.shape) == 3:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_bgr.copy()

    # Resize to 48x48 for CNN
    face_gray = cv2.resize(gray, (48,48))

    # Resize to 64x64 for classical features
    face_classical = cv2.resize(face_gray, (64,64))

    return face_gray, face_classical

