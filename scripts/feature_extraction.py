import numpy as np
import cv2
import os


def calculate_lbp(image):
    lbp_image = np.zeros_like(image)
    for row in range(1, image.shape[0] - 1):
        for col in range(1, image.shape[1] - 1):
            center = image[row, col]
            binary_pattern = [image[row - 1, col - 1] >= center, image[row - 1, col] >= center,
                              image[row - 1, col + 1] >= center,
                              image[row, col + 1] >= center, image[row + 1, col + 1] >= center,
                              image[row + 1, col] >= center,
                              image[row + 1, col - 1] >= center, image[row, col - 1] >= center]
            lbp_image[row, col] = sum([val * (1 << i) for i, val in enumerate(binary_pattern)])
    return lbp_image


def apply_dct(image):
    return cv2.dct(np.float32(image))


def extract_features(images, start_idx=0, end_idx=None):
    if end_idx is None:
        end_idx = len(images)
    features = []
    for idx in range(start_idx, end_idx):
        print(f"Processing image {idx + 1}/{len(images)}")
        lbp_image = calculate_lbp(images[idx])
        dct_image = apply_dct(lbp_image)
        mean_val = np.mean(dct_image)
        features.append(mean_val)

    return np.array(features).reshape(-1, 1)  # 2D-sé alakítjuk a kinyert jellemzőket


def save_features(features, labels, output_file):
    np.savez(output_file, features=features, labels=labels)
    print(f"Features saved to {output_file}")


def load_features(file_path):
    data = np.load(file_path)
    return data['features'], data['labels']


if __name__ == "__main__":
    from preprocess import preprocess_images

    revised_dir = os.path.abspath('data/CASIA2.0_revised')
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)
    output_file = os.path.join(result_dir, 'features_labels.npz')

    if os.path.exists(output_file):
        print(f"Loading features from {output_file}")
        all_features, labels = load_features(output_file)
    else:
        images, labels = preprocess_images(revised_dir)
        print(f"Images loaded: {len(images)}")

        # Képek feldolgozása részenként
        batch_size = 1000
        all_features = []
        for i in range(0, len(images), batch_size):
            print(f"Processing batch {i + 1} to {min(i + batch_size, len(images))}")
            features = extract_features(images, start_idx=i, end_idx=min(i + batch_size, len(images)))
            all_features.extend(features)

        save_features(all_features, labels, output_file)

    print(f"Feature extraction completed. Number of features: {len(all_features)}")
