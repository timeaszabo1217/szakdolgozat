import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from preprocess import preprocess_images
from feature_extraction import extract_features
from train_classifier import load_classifier


def convert_to_ycbcr(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Kép betöltése
    if image is None:
        print(f"Could not read image: {image_path}")
        return None
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)  # Kép konvertálása YCbCr színtérbe
    ycbcr_image = cv2.cvtColor(ycbcr_image, cv2.COLOR_BGR2GRAY)  # Konvertálás szürkeárnyalatossá (1 csatorna)
    return ycbcr_image


def preprocess_images(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            print(f"Processing file: {img_path}")
            ycbcr_image = convert_to_ycbcr(img_path)
            if ycbcr_image is not None:
                ycbcr_image = ycbcr_image.astype(np.float32)  # Konvertálás lebegőpontos típusra
                images.append(ycbcr_image)
                if 'Au' in filename:
                    labels.append(0)  # Autentikus
                elif 'Tp' in filename:
                    labels.append(1)  # Hamisított
    return images, labels


def calculate_lbp(image):
    lbp_image = np.zeros(image.shape, dtype=np.float32)
    for row in range(1, image.shape[0] - 1):
        for col in range(1, image.shape[1] - 1):
            center = image[row, col]
            binary_string = ''
            binary_string += '1' if image[row - 1, col - 1] >= center else '0'
            binary_string += '1' if image[row - 1, col] >= center else '0'
            binary_string += '1' if image[row - 1, col + 1] >= center else '0'
            binary_string += '1' if image[row, col + 1] >= center else '0'
            binary_string += '1' if image[row + 1, col + 1] >= center else '0'
            binary_string += '1' if image[row + 1, col] >= center else '0'
            binary_string += '1' if image[row + 1, col - 1] >= center else '0'
            binary_string += '1' if image[row, col - 1] >= center else '0'
            lbp_image[row, col] = int(binary_string, 2)
    return lbp_image


def apply_dct(image):
    return cv2.dct(image)


def test_classifier(new_dataset_dir, classifier_file):
    # Előkészítjük az új adatkészlet képeit
    images, labels = preprocess_images(new_dataset_dir)
    features = extract_features(images)

    # Betöltjük a mentett osztályozót
    classifier = load_classifier(classifier_file)

    # Osztályozás és értékelés
    predictions = classifier.predict(features)
    report = classification_report(labels, predictions, target_names=['Authentic', 'Tampered'])
    print(report)

    # Konfúziós mátrix megjelenítése
    ConfusionMatrixDisplay.from_estimator(classifier, features, labels, display_labels=['Authentic', 'Tampered'])
    plt.show()

    return report


if __name__ == "__main__":
    # Az új adatkészlet könyvtárának megadása
    new_dataset_dir = os.path.abspath('../data/CASIA2.0_Groundtruth')

    # A mentett osztályozó fájl elérési útja
    classifier_file = os.path.join('results', 'classifier_model.pkl')

    # Teszteljük az osztályozót az új adatkészleten
    test_report = test_classifier(new_dataset_dir, classifier_file)

    # Az eredmény hozzáfűzése a meglévő fájlhoz
    result_file = os.path.join('results', 'test_results.txt')
    with open(result_file, 'a') as f:  # 'a' mód (append) hozzáfűzéshez
        f.write("\n\n")  # Üres sorok az új eredmények előtt
        f.write(test_report)
    print(f"Test results saved to {result_file}")
