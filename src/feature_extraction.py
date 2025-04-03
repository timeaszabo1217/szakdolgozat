import os
import cv2
import numpy as np
from preprocess import load_preprocessed_data, split_into_overlapping_blocks, apply_fft


def calculate_lbp(image):
    image = image.astype(np.float32)
    lbp_image = np.zeros_like(image, dtype=np.float32)

    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            g_t = image[x, y]
            binary_pattern = []

            neighbors = [
                image[x - 1, y - 1], image[x - 1, y], image[x - 1, y + 1],
                image[x, y + 1], image[x + 1, y + 1], image[x + 1, y],
                image[x + 1, y - 1], image[x, y - 1]
            ]

            for g_x in neighbors:
                binary_pattern.append(1 if g_x >= g_t else 0)

            lbp_image[x, y] = sum([binary_pattern[i] * (2 ** i) for i in range(8)])  # type: ignore

    return lbp_image


def calculate_ltp(image, th=5):
    image = image.astype(np.float32)
    ltp_image = np.zeros_like(image, dtype=np.float32)

    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            g_t = image[x, y]
            ternary_pattern = []

            neighbors = [
                image[x - 1, y - 1], image[x - 1, y], image[x - 1, y + 1],
                image[x, y + 1], image[x + 1, y + 1], image[x + 1, y],
                image[x + 1, y - 1], image[x, y - 1]
            ]

            for g_x in neighbors:
                if g_x >= g_t + th:
                    ternary_pattern.append(1)
                elif g_x <= g_t - th:
                    ternary_pattern.append(-1)
                else:
                    ternary_pattern.append(0)

            ltp_image[x, y] = sum([ternary_pattern[i] * (3 ** i) for i in range(8)])  # type: ignore

    return ltp_image


def calculate_eltp(image):
    image = image.astype(np.float32)

    g_t_e = np.mean(image)
    t_e = np.mean(np.abs(image - g_t_e))

    def s_e(g_x, g_t_e, t_e):
        return np.where(g_x >= g_t_e + t_e, 1, np.where(g_x <= g_t_e - t_e, -1, 0))

    s_matrix = s_e(image, g_t_e, t_e)

    eltp_x = np.sum(s_matrix == 1, axis=None)
    eltp_n = np.sum(s_matrix == -1, axis=None)

    eltp = eltp_x * (image.shape[1] + 2) - 4 * (eltp_x * (eltp_x + 1)) / 2 + eltp_n

    return eltp


def extract_features(images, method, batch_size=200):
    features = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_features = []

        print(f"Processing batch {i // batch_size + 1} of {len(images) // batch_size + 1}")

        for j, image in enumerate(batch_images):
            print(f"Processing image {i + j + 1} of {len(images)}")

            image_features = []
            blocks = split_into_overlapping_blocks(image)

            if method == 'lbp':
                lbp_blocks = [calculate_lbp(block) for block in blocks]
                lbp_dct_features = [np.mean(cv2.dct(np.float32(lbp))) for lbp in lbp_blocks]
                image_features.append(np.mean(lbp_dct_features))

            elif method == 'ltp':
                ltp_blocks = [calculate_ltp(block) for block in blocks]
                ltp_dct_features = [np.mean(cv2.dct(np.float32(ltp))) for ltp in ltp_blocks]
                image_features.append(np.mean(ltp_dct_features))

            elif method == 'fft_eltp':
                fft_image = apply_fft(image)
                fft_blocks = split_into_overlapping_blocks(fft_image)
                eltp_blocks = [calculate_eltp(block) for block in fft_blocks]
                eltp_features = [np.mean(eltp) for eltp in eltp_blocks]
                image_features.append(np.mean(eltp_features))

            batch_features.append(np.array(image_features))

        features.append(np.array(batch_features))

    return np.concatenate(features, axis=0)


def save_features(features, labels, output_file):
    np.savez(output_file, features=features, labels=labels)
    print(f"Features saved to {output_file}")


def load_features(file_path):
    data = np.load(file_path)
    return data['features'], data['labels']


if __name__ == "__main__":
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)

    preprocessed_data = os.path.join(result_dir, 'preprocessed_data.npz')
    images, labels = load_preprocessed_data(preprocessed_data)

    output_files = {
        'lbp': os.path.join(result_dir, 'lbp_features_labels.npz'),
        'ltp': os.path.join(result_dir, 'ltp_features_labels.npz'),
        'fft_eltp': os.path.join(result_dir, 'fft_eltp_features_labels.npz')
    }

    for method in ['lbp', 'ltp', 'fft_eltp']:
        if os.path.exists(output_files[method]):
            print(f"Loading {method.upper()} features from {output_files[method]}")
            features, labels = load_features(output_files[method])
        else:
            print(f"Extracting {method.upper()} features")
            features = extract_features(images, method, batch_size=200)
            save_features(features, labels, output_files[method])
