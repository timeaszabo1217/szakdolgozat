import os
import cv2
import numpy as np


def convert_to_ycbcr(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return ycbcr_image[:, :, 1]  # Csak a Cb (krominancia) komponens kivétele


def preprocess_images(revised_dir):
    images = []
    labels = []

    # Ellenőrizzük az elérési utakat
    print(f"Checking directory: {revised_dir}")
    for subdir in ['Au', 'Tp']:
        full_dir = os.path.join(revised_dir, subdir)
        print(f"Checking subdirectory: {full_dir}")
        if not os.path.exists(full_dir):
            print(f"Directory not found: {full_dir}")
            continue
        for filename in os.listdir(full_dir):
            if filename.endswith('.jpg'):
                file_path = os.path.join(full_dir, filename)
                print(f"Processing Revised file: {file_path}")
                ycbcr_image = convert_to_ycbcr(file_path)
                if ycbcr_image is not None:
                    images.append(ycbcr_image)
                    labels.append(0 if subdir == 'Au' else 1)  # Au képek eredetiek, Tp képek hamisítottak

    return images, labels


if __name__ == "__main__":
    revised_dir = os.path.abspath('../data/CASIA2.0_revised')
    print(f"Using revised directory: {revised_dir}")
    images, labels = preprocess_images(revised_dir)
    print(f"Number of loaded images: {len(images)}")
