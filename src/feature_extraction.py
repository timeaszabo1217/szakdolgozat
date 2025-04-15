import os
import joblib
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


def extract_features(images, labels, methods, components, output_file_base, batch_size=200):
    all_features = []

    for method in methods:
        for comp in components:
            print(f"Extracting {method.upper()} features from {comp} component")

            output_file = None
            if output_file_base is not None:
                output_file = output_file_base.replace('.joblib', f'_{comp}.joblib')

            if os.path.exists(output_file):
                print(f"Features for {method.upper()} ({comp}) already exist. Skipping extraction.")
                continue

            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                batch_features = []

                print(f"Processing batch {i // batch_size + 1} of {len(images) // batch_size + 1}")

                for j, image in enumerate(batch_images):
                    print(f"Processing image {i + j + 1} of {len(images)}")

                    Cb, Cr = image
                    if comp == 'CbCr':
                        image = (Cb.astype(np.float32) + Cr.astype(np.float32)) / 2
                    elif comp == 'Cb':
                        image = Cb
                    elif comp == 'Cr':
                        image = Cr

                    image_features = []

                    if method == 'lbp':
                        blocks = split_into_overlapping_blocks(image, block_size=(3, 3))
                        lbp_blocks = [calculate_lbp(block) for block in blocks]
                        lbp_features = [cv2.dct(lbp) for lbp in lbp_blocks]
                        image_features.append(np.mean(lbp_features))

                    elif method == 'ltp':
                        blocks = split_into_overlapping_blocks(image, block_size=(3, 3))
                        ltp_blocks = [calculate_ltp(block) for block in blocks]
                        ltp_features = [cv2.dct(ltp) for ltp in ltp_blocks]
                        image_features.append(np.mean(ltp_features))

                    elif method == 'fft_eltp':
                        fft_image = apply_fft(image)
                        blocks = split_into_overlapping_blocks(fft_image, block_size=(3, 3))
                        eltp_features = [calculate_eltp(block) for block in blocks]
                        image_features.append(np.mean(eltp_features))

                    if image_features:
                        batch_features.append(image_features)
                    else:
                        print(f"Warning: No valid features found for image {i + j + 1}.")

                if batch_features:
                    all_features.extend(batch_features)
                    save_features(batch_features, batch_labels, output_file, append=(i > 0))
                else:
                    print(f"Warning: No valid features in batch {i // batch_size + 1}.")

    return np.array(all_features)


def save_features(batch_features, batch_labels, output_file, append=False):
    mode = 'ab' if append else 'wb'
    saved = 0

    with open(output_file, mode) as file:
        for feature, label in zip(batch_features, batch_labels):
            joblib.dump({'feature': feature, 'label': label}, file)
            saved += 1
    print(f"Saved {saved} new samples to {output_file}")


def load_features(file_path):
    features = []
    labels = []
    count = 0

    with open(file_path, 'rb') as file:
        while True:
            try:
                entry = joblib.load(file)
                features.append(entry['feature'])
                labels.append(entry['label'])
                count += 1
                if count % 200 == 0:
                    print(f"Loaded {count} feature")
            except EOFError:
                break

    print(f"Loaded {len(features)} feature")
    return features, labels


if __name__ == "__main__":
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)

    preprocessed_data = os.path.join(result_dir, 'preprocessed_data.joblib')
    images, labels = load_preprocessed_data(preprocessed_data)

    methods = ['lbp', 'ltp', 'fft_eltp']
    components = ['CbCr', 'Cb', 'Cr']

    for method in methods:
        output_file_base = os.path.join(result_dir, f"{method}_features_labels.joblib")
        extract_features(images, labels, [method], components, output_file_base, batch_size=200)
