import os

# Mapear nomes das pastas → labels FER2013
EMOTION_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}

def load_fer_dataset(root):
    image_paths = []
    labels = []

    for label_name in os.listdir(root):
        label_path = os.path.join(root, label_name)
        if not os.path.isdir(label_path):
            continue

        # Verificar se o nome da pasta está no dicionário
        if label_name not in EMOTION_MAP:
            print(f"[WARNING] Ignoring folder: {label_name}")
            continue

        numeric_label = EMOTION_MAP[label_name]

        for file in os.listdir(label_path):
            if file.lower().endswith((".jpg", ".png")):
                image_paths.append(os.path.join(label_path, file))
                labels.append(numeric_label)

    return image_paths, labels
