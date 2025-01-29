import os
import cv2
import numpy as np
from preprocess import preprocess_images


def calculate_lbp(image):
    image = image.astype(np.float32)
    lbp_image = np.zeros_like(image, dtype=np.float32)
    for row in range(1, image.shape[0] - 1):
        for col in range(1, image.shape[1] - 1):
            center = image[row, col]
            binary_pattern = [
                int(image[row - 1, col - 1] >= center),
                int(image[row - 1, col] >= center),
                int(image[row - 1, col + 1] >= center),
                int(image[row, col + 1] >= center),
                int(image[row + 1, col + 1] >= center),
                int(image[row + 1, col] >= center),
                int(image[row + 1, col - 1] >= center),
                int(image[row, col - 1] >= center)
            ]
            lbp_image[row, col] = sum([val * (1 << i) for i, val in enumerate(binary_pattern)])
    return lbp_image


def calculate_ltp(image):
    image = image.astype(np.float32)
    ltp_image = np.zeros_like(image, dtype=np.float32)
    threshold = 5
    for row in range(1, image.shape[0] - 1):
        for col in range(1, image.shape[1] - 1):
            center = image[row, col]
            binary_pattern = []
            for r in [-1, 0, 1]:
                for c in [-1, 0, 1]:
                    if r == 0 and c == 0:
                        continue
                    neighbor = image[row + r, col + c]
                    if neighbor > center + threshold:
                        binary_pattern.append(1)
                    elif neighbor < center - threshold:
                        binary_pattern.append(-1)
                    else:
                        binary_pattern.append(0)
            ltp_image[row, col] = sum([val * (3 ** i) for i, val in enumerate(binary_pattern)])
    return ltp_image


def calculate_eltp(image):
    image = np.real(image).astype(np.float32)
    eltp_image = np.zeros_like(image, dtype=np.float32)
    for row in range(1, image.shape[0] - 1):
        for col in range(1, image.shape[1] - 1):
            center = image[row, col]
            binary_pattern = [
                int(image[row - 1, col - 1] >= center),
                int(image[row - 1, col] >= center),
                int(image[row - 1, col + 1] >= center),
                int(image[row, col + 1] >= center),
                int(image[row + 1, col + 1] >= center),
                int(image[row + 1, col] >= center),
                int(image[row + 1, col - 1] >= center),
                int(image[row, col - 1] >= center)
            ]
            eltp_image[row, col] = sum([val * (1 << i) for i, val in enumerate(binary_pattern)])
    return eltp_image


def apply_dct(image):
    return cv2.dct(np.float32(image))


def extract_features_lbp_ltp(images, start_idx=0, end_idx=None):
    if end_idx is None:
        end_idx = len(images)
    features = []
    for idx in range(start_idx, end_idx):
        print(f"Processing image {idx + 1}/{end_idx} for LBP-LTP")
        lbp_image = calculate_lbp(images[idx])
        ltp_image = calculate_ltp(images[idx])
        dct_lbp = apply_dct(lbp_image)
        dct_ltp = apply_dct(ltp_image)
        mean_val_lbp = np.mean(dct_lbp)
        mean_val_ltp = np.mean(dct_ltp)
        combined_features = np.array([mean_val_lbp, mean_val_ltp])
        features.append(combined_features)
    print(f"Extracted features from {start_idx} to {end_idx}: {len(features)}")
    return np.array(features)


def extract_features_fft_eltp(images):
    features = []
    for idx, image in enumerate(images):
        print(f"Processing image {idx + 1}/{len(images)} for FFT-ELTP")
        fft_image = np.fft.fft2(image)
        eltp_image = calculate_eltp(fft_image)
        dct_eltp = apply_dct(eltp_image)
        mean_val_eltp = np.mean(dct_eltp)
        features.append(mean_val_eltp)
    print(f"Extracted features: {len(features)}")
    return np.array(features)


def save_features(features, labels, output_file):
    np.savez(output_file, features=features, labels=labels)
    print(f"Features saved to {output_file}")


def load_features(file_path):
    data = np.load(file_path)
    return data['features'], data['labels']


if __name__ == "__main__":
    revised_dir = os.path.abspath('../data/CASIA2.0_revised')
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)
    output_file_lbp_ltp = os.path.join(result_dir, 'lbp_ltp_features_labels.npz')
    output_file_fft_eltp = os.path.join(result_dir, 'fft_eltp_features_labels.npz')

    # LBP és LTP jellemzők feldolgozása és mentése
    if os.path.exists(output_file_lbp_ltp):
        print(f"Loading LBP and LTP features from {output_file_lbp_ltp}")
        lbp_ltp_features, labels = load_features(output_file_lbp_ltp)
    else:
        images, labels = preprocess_images(revised_dir)
        print(f"Images loaded: {len(images)}")
        # Képek feldolgozása részenként
        batch_size = 1000
        lbp_ltp_features = []
        for i in range(0, len(images), batch_size):
            print(f"Processing batch {i + 1} to {min(i + batch_size, len(images))} for LBP-LTP")
            features = extract_features_lbp_ltp(images, start_idx=i, end_idx=min(i + batch_size, len(images)))
            lbp_ltp_features.extend(features)

        lbp_ltp_features = np.array(lbp_ltp_features)
        save_features(lbp_ltp_features, labels, output_file_lbp_ltp)

    print(f"LBP-LTP feature extraction completed. Number of features: {len(lbp_ltp_features)}")

    # FFT és ELTP jellemzők feldolgozása és mentése
    if os.path.exists(output_file_fft_eltp):
        print(f"Loading FFT and ELTP features from {output_file_fft_eltp}")
        fft_eltp_features, labels = load_features(output_file_fft_eltp)
    else:
        images, labels = preprocess_images(revised_dir)
        print(f"Images loaded: {len(images)}")
        # Képek feldolgozása részenként
        batch_size = 500
        fft_eltp_features = []
        for i in range(0, len(images), batch_size):
            print(f"Processing batch {i + 1} to {min(i + batch_size, len(images))} for FFT-ELTP")
            features = extract_features_fft_eltp(images[i:min(i + batch_size, len(images))])
            fft_eltp_features.extend(features)

        fft_eltp_features = np.array(fft_eltp_features)
        save_features(fft_eltp_features, labels, output_file_fft_eltp)

    print(f"FFT-ELTP feature extraction completed. Number of features: {len(fft_eltp_features)}")
