import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from preprocess import preprocess_images
from feature_extraction import extract_features
from train_classifier import load_classifier


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
    # Az új adatkészlet könyvtárának megadása (cseréld le a megfelelő elérési útra)
    new_dataset_dir = os.path.abspath('data/CASIA2.0_Groundtruth')

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
