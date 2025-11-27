import kagglehub
import shutil
import os

#1. Download the dataset via Kaggle Hub.
path = kagglehub.dataset_download("msambare/fer2013")
print("Dataset downloaded to:", path)

#2. Destination path in your project
dest = "./data/fer2013"

# 3. Create folder if it doesn't exist
os.makedirs(dest, exist_ok=True)

#4. Copy EVERYTHING from the path to the folder ./data/fer2013
for item in os.listdir(path):
    source = os.path.join(path, item)
    target = os.path.join(dest, item)

    if os.path.isdir(source):
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)

print("Dataset copied to:", dest)
