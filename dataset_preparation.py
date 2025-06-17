
import os
import cv2
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

SOURCE_DIR = "Final_Training/Images"
OUTPUT_TRAIN = "GTSRB_cleaned/train"
OUTPUT_VAL = "GTSRB_cleaned/val"
SPLIT_RATIO = 0.8

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def crop_and_save():
    all_images = []
    all_labels = []

    for class_folder in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_folder)
        if not os.path.isdir(class_path): continue

        csv_file = os.path.join(class_path, f"GT-{class_folder}.csv")
        if not os.path.exists(csv_file): continue

        df = pd.read_csv(csv_file, sep=';')
        for _, row in df.iterrows():
            filename = row['Filename']
            img_path = os.path.join(class_path, filename)

            img = cv2.imread(img_path)
            if img is None: continue

            # Crop the image using ROI
            x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
            cropped_img = img[y1:y2, x1:x2]
            label = str(row['ClassId'])

            all_images.append((cropped_img, label))

    print(f"Total images collected: {len(all_images)}")

    # Split into train and validation
    train_data, val_data = train_test_split(all_images, test_size=(1 - SPLIT_RATIO), stratify=[lbl for _, lbl in all_images], random_state=42)

    for cropped_img, label in train_data:
        save_path = os.path.join(OUTPUT_TRAIN, label)
        ensure_dir(save_path)
        fname = f"{label}_{len(os.listdir(save_path))}.png"
        cv2.imwrite(os.path.join(save_path, fname), cv2.resize(cropped_img, (160, 160)))

    for cropped_img, label in val_data:
        save_path = os.path.join(OUTPUT_VAL, label)
        ensure_dir(save_path)
        fname = f"{label}_{len(os.listdir(save_path))}.png"
        cv2.imwrite(os.path.join(save_path, fname), cv2.resize(cropped_img, (160, 160)))

    print("Cropping and dataset preparation complete!")
    print(f"Train directory: {OUTPUT_TRAIN}")
    print(f"Validation directory: {OUTPUT_VAL}")

if __name__ == "__main__":
    crop_and_save()
