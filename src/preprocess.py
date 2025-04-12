import os
import joblib
import cv2
import numpy as np
from numpy.fft import fft2, fftshift


def convert_to_ycbcr(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    h, w = image.shape[:2]
    if w > h:
        image = cv2.transpose(image)
        image = cv2.flip(image, flipCode=1)

    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return ycbcr_image


def get_chrominance_component(image):
    ycbcr_image = convert_to_ycbcr(image)
    Y, Cb, Cr = cv2.split(ycbcr_image)
    return Cb


def apply_fft(image):
    fft_image = fft2(image)
    fft_image_shifted = fftshift(fft_image)
    fft_magnitude = np.abs(fft_image_shifted)
    return fft_magnitude


def split_into_overlapping_blocks(image, block_size=(3, 3), overlap=1):
    blocks = []
    h, w = image.shape
    block_h, block_w = block_size

    for i in range(0, h - block_h + 1, block_h - overlap):
        for j in range(0, w - block_w + 1, block_w - overlap):
            block = image[i:i + block_h, j:j + block_w]
            blocks.append(block)

    if h % block_h != 0:
        for j in range(0, w - block_w + 1, block_w - overlap):
            block = image[h - block_h:h, j:j + block_w]
            blocks.append(block)

    if w % block_w != 0:
        for i in range(0, h - block_h + 1, block_h - overlap):
            block = image[i:i + block_h, w - block_w:w]
            blocks.append(block)

    if h % block_h != 0 and w % block_w != 0:
        block = image[h - block_h:h, w - block_w:w]
        blocks.append(block)

    return blocks


def preprocess_images(image_dir):
    images, labels = [], []

    subdirs = ['Au', 'Tp']
    subdir_exists = any(subdir in os.listdir(image_dir) for subdir in subdirs)

    for subdir in (subdirs if subdir_exists else [image_dir]):
        for file in os.listdir(os.path.join(image_dir, subdir) if subdir_exists else [image_dir]):
            if file.endswith(('.jpg', '.png', '.tif')):
                file_path = os.path.join(image_dir, subdir, file) if subdir_exists else os.path.join(image_dir, file)
                print(f"Processing file: {file_path}")
                cb_component = get_chrominance_component(file_path)
                if cb_component is not None:
                    images.append(cb_component)
                    label = 0 if 'Au' in (subdir if subdir_exists else file) else 1
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
    revised_dir = os.path.abspath('../data/CASIA2.0_revised')
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)

    output_file = os.path.join(result_dir, 'preprocessed_data.joblib')

    if os.path.exists(output_file):
        print("Preprocessed data already exists. Skipping preprocessing.")
    else:
        print("Processing images")
        images, labels = preprocess_images(revised_dir)
        save_preprocessed_data(images, labels, output_file)
