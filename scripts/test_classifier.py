import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from preprocess import preprocess_images
from feature_extraction import extract_features_lbp_ltp, extract_features_fft_eltp
from train_classifier import load_classifier


def test_classifier(new_dataset_dir, classifier_file_lbp_ltp, classifier_file_fft_eltp):
    # Előkészítjük az új adatkészlet képeit
    images, labels = preprocess_images(new_dataset_dir)
    print(f"Number of loaded images: {len(images)}")
    print(f"Number of labels loaded: {len(labels)}")

    # LBP-LTP jellemzők kinyerése
    features_lbp_ltp = extract_features_lbp_ltp(images)
    print(f"Number of LBP-LTP features extracted: {features_lbp_ltp.shape[0]}")

    # Ellenőrizzük, hogy a features_lbp_ltp nem üres-e
    if features_lbp_ltp.size == 0:
        raise ValueError("Extracted LBP-LTP features are empty. Please check the feature extraction process.")

    # Betöltjük a mentett LBP-LTP osztályozót
    classifier_lbp_ltp = load_classifier(classifier_file_lbp_ltp)

    # LBP-LTP osztályozás és értékelés
    predictions_lbp_ltp = classifier_lbp_ltp.predict(features_lbp_ltp)
    report_lbp_ltp = classification_report(labels, predictions_lbp_ltp, target_names=['Authentic', 'Tampered'])
    print("LBP-LTP Classification Report:\n", report_lbp_ltp)

    # Konfúziós mátrix megjelenítése LBP-LTP
    ConfusionMatrixDisplay.from_estimator(classifier_lbp_ltp, features_lbp_ltp, labels, display_labels=['Authentic', 'Tampered'])
    plt.title('Confusion Matrix - LBP-LTP')
    plt.show()

    # FFT-ELTP jellemzők kinyerése
    features_fft_eltp = extract_features_fft_eltp(images)
    print(f"Number of FFT-ELTP features extracted: {features_fft_eltp.shape[0]}")

    # Ellenőrizzük, hogy a features_fft_eltp nem üres-e
    if features_fft_eltp.size == 0:
        raise ValueError("Extracted FFT-ELTP features are empty. Please check the feature extraction process.")

    # Betöltjük a mentett FFT-ELTP osztályozót
    classifier_fft_eltp = load_classifier(classifier_file_fft_eltp)

    # FFT-ELTP osztályozás és értékelés
    predictions_fft_eltp = classifier_fft_eltp.predict(features_fft_eltp)
    report_fft_eltp = classification_report(labels, predictions_fft_eltp, target_names=['Authentic', 'Tampered'])
    print("FFT-ELTP Classification Report:\n", report_fft_eltp)

    # Konfúziós mátrix megjelenítése FFT-ELTP
    ConfusionMatrixDisplay.from_estimator(classifier_fft_eltp, features_fft_eltp, labels, display_labels=['Authentic', 'Tampered'])
    plt.title('Confusion Matrix - FFT-ELTP')
    plt.show()

    return report_lbp_ltp, report_fft_eltp


if __name__ == "__main__":
    # Az új adatkészlet könyvtárának megadása
    new_dataset_dir = os.path.abspath('../data/CASIA2.0_Groundtruth')

    # A mentett osztályozó fájl elérési útja
    classifier_file_lbp_ltp = os.path.join('results', 'classifier_lbp_ltp.pkl')
    classifier_file_fft_eltp = os.path.join('results', 'classifier_fft_eltp.pkl')

    # Teszteljük az osztályozókat az új adatkészleten
    report_lbp_ltp, report_fft_eltp = test_classifier(new_dataset_dir, classifier_file_lbp_ltp, classifier_file_fft_eltp)

    # Az eredmények hozzáfűzése a meglévő fájlhoz
    result_file = os.path.join('results', 'test_results.txt')
    with open(result_file, 'a') as f:
        f.write("\n\nLBP-LTP Classification Report:\n")
        f.write(report_lbp_ltp)
        f.write("\nFFT-ELTP Classification Report:\n")
        f.write(report_fft_eltp)
    print(f"Test results saved to {result_file}")
