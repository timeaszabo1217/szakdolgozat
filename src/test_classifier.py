import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from preprocess import preprocess_images
from feature_extraction import extract_features_lbp_ltp, extract_features_fft_eltp
from train_classifier import load_classifier


def test_classifier(new_dataset_dir, classifier_file_lbp_ltp, classifier_file_fft_eltp):
    images, labels = preprocess_images(new_dataset_dir)
    print(f"Number of loaded images: {len(images)}")
    print(f"Number of labels loaded: {len(labels)}")

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Unique labels and their counts:", dict(zip(unique_labels, counts)))

    if len(unique_labels) < 2:
        raise ValueError(
            "Only one class is present in the dataset. Please ensure both 'Authentic' and 'Tampered' exist.")

    # LBP-LTP jellemzők kinyerése
    features_lbp_ltp = extract_features_lbp_ltp(images)
    print(f"Number of LBP-LTP features extracted: {features_lbp_ltp.shape}")

    if features_lbp_ltp.size == 0:
        raise ValueError("Extracted LBP-LTP features are empty. Please check the feature extraction process.")

    classifier_lbp_ltp = load_classifier(classifier_file_lbp_ltp)
    print("LBP-LTP Classifier parameters:", classifier_lbp_ltp.get_params())

    predictions_lbp_ltp = classifier_lbp_ltp.predict(features_lbp_ltp)
    report_lbp_ltp = classification_report(labels, predictions_lbp_ltp, target_names=['Authentic', 'Tampered'],
                                           zero_division=1)
    print("LBP-LTP Classification Report:\n", report_lbp_ltp)

    ConfusionMatrixDisplay.from_estimator(classifier_lbp_ltp, features_lbp_ltp, labels,
                                          display_labels=['Authentic', 'Tampered'])
    plt.title('Confusion Matrix - LBP-LTP')
    plt.show()

    # FFT-ELTP jellemzők kinyerése
    features_fft_eltp = extract_features_fft_eltp(images)
    print(f"Number of FFT-ELTP features extracted: {features_fft_eltp.shape}")

    if features_fft_eltp.size == 0:
        raise ValueError("Extracted FFT-ELTP features are empty. Please check the feature extraction process.")

    if features_fft_eltp.ndim == 1:
        features_fft_eltp = features_fft_eltp.reshape(-1, 1)

    classifier_fft_eltp = load_classifier(classifier_file_fft_eltp)
    print("FFT-ELTP Classifier parameters:", classifier_fft_eltp.get_params())

    predictions_fft_eltp = classifier_fft_eltp.predict(features_fft_eltp)
    report_fft_eltp = classification_report(labels, predictions_fft_eltp, target_names=['Authentic', 'Tampered'],
                                            zero_division=1)
    print("FFT-ELTP Classification Report:\n", report_fft_eltp)

    ConfusionMatrixDisplay.from_estimator(classifier_fft_eltp, features_fft_eltp, labels,
                                          display_labels=['Authentic', 'Tampered'])
    plt.title('Confusion Matrix - FFT-ELTP')
    plt.show()

    return report_lbp_ltp, report_fft_eltp


if __name__ == "__main__":
    new_dataset_dir = os.path.abspath('../data/CASIA2.0_test')
    classifier_file_lbp_ltp = os.path.join('results', 'classifier_lbp_ltp.pkl')
    classifier_file_fft_eltp = os.path.join('results', 'classifier_fft_eltp.pkl')

    report_lbp_ltp, report_fft_eltp = test_classifier(new_dataset_dir, classifier_file_lbp_ltp,
                                                      classifier_file_fft_eltp)

    result_file = os.path.join('results', 'test_results.txt')
    with open(result_file, 'w', encoding="utf-8") as f:
        f.write("LBP-LTP\n")
        f.write(report_lbp_ltp)
        f.write("\n\nFFT-ELTP\n")
        f.write(report_fft_eltp)

    print(f"Test results saved to {result_file}")
