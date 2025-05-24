import os
import joblib
import cv2
import numpy as np
from numpy.fft import fft2, fftshift


def convert_to_ycbcr(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    try:
        ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    except Exception as e:
        print(f"Warning: Could not convert image {image_path} to YCrCb: {e}")
        return None
    return ycbcr_image


def get_chrominance_components(image_path):
    ycbcr_image = convert_to_ycbcr(image_path)
    if ycbcr_image is None:
        return None, None
    try:
        Y, Cb, Cr = cv2.split(ycbcr_image)
    except Exception as e:
        print(f"Warning: Could not split YCrCb channels for {image_path}: {e}")
        return None, None
    return Cb, Cr


def apply_fft(image):
    fft_image = fft2(image)
    fft_image_shifted = fftshift(fft_image)
    fft_magnitude = np.abs(fft_image_shifted)

    log_magnitude = np.log1p(fft_magnitude)
    max_val = np.max(log_magnitude)

    if max_val == 0:
        return log_magnitude
    return log_magnitude / max_val


def split_into_overlapping_blocks(image, block_size=(3, 3), num_blocks=(254, 382)):  # 97028
    h, w = image.shape
    block_h, block_w = block_size
    num_blocks_h, num_blocks_w = num_blocks

    step_h = (h - block_h) / (num_blocks_h - 1) if num_blocks_h > 1 else 1
    step_w = (w - block_w) / (num_blocks_w - 1) if num_blocks_w > 1 else 1

    blocks = []
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            start_i = int(round(i * step_h))
            start_j = int(round(j * step_w))

            if start_i + block_h > h:
                start_i = h - block_h
            if start_j + block_w > w:
                start_j = w - block_w

            block = image[start_i:start_i + block_h, start_j:start_j + block_w]
            blocks.append(block)

    return blocks


def preprocess_images(image_dir):
    images, labels = [], []

    subdirs = ['Au', 'Tp']
    subdir_exists = all(os.path.isdir(os.path.join(image_dir, subdir)) for subdir in subdirs)

    if subdir_exists:
        for subdir in subdirs:
            folder_path = os.path.join(image_dir, subdir)
            files = os.listdir(folder_path)
            for file in files:
                if file.lower().endswith('.jpg'):  # (('.jpeg', '.png', '.tif'))
                    file_path = os.path.join(folder_path, file)
                    print(f"Processing file: {file_path}")
                    Cb, Cr = get_chrominance_components(file_path)
                    if Cb is not None and Cr is not None:
                        images.append((Cb, Cr))
                        label = 0 if subdir == 'Au' else 1
                        labels.append(label)
    else:
        files = os.listdir(image_dir)
        for file in files:
            if file.lower().endswith('.jpg'):  # (('.jpeg', '.png', '.tif'))
                file_path = os.path.join(image_dir, file)
                print(f"Processing file: {file_path}")
                Cb, Cr = get_chrominance_components(file_path)
                if Cb is not None and Cr is not None:
                    images.append((Cb, Cr))
                    label = 0 if 'Au' in file else 1
                    labels.append(label)

    print(f"Number of images: {len(images)}, Number of labels: {len(labels)}")
    return images, labels


def save_preprocessed_data(images, labels, output_file):
    joblib.dump({'images': images, 'labels': labels}, output_file)
    print(f"Processed data saved to {output_file}")


def load_preprocessed_data(file_path):
    data = joblib.load(file_path)
    print(f"Loaded {len(data['images'])} images from {file_path}")
    return data['images'], data['labels']


if __name__ == "__main__":
    data_dir = os.path.abspath('../data/CASIA1.0')
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, 'preprocessed_data.joblib')

    if os.path.exists(output_file):
        print("Preprocessed data already exists. Skipping preprocessing.")
    else:
        print("Processing images")
        images, labels = preprocess_images(data_dir)
        save_preprocessed_data(images, labels, output_file)
