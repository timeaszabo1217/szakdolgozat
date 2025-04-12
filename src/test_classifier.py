import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, accuracy_score, recall_score
from preprocess import preprocess_images
from feature_extraction import extract_features
from train_classifier import load_classifier


def evaluate_classifier(method, images, labels, classifier_file):
    features = extract_features(images, labels, method=method, output_file=None)
    print(f"Number of {method.upper()} features extracted: {features.shape}")

    if features.size == 0:
        raise ValueError(f"Extracted {method.upper()} features are empty.")

    classifier = load_classifier(classifier_file)
    print(f"{method.upper()} Classifier parameters: ", classifier.get_params())

    predictions = classifier.predict(features)

    report = classification_report(labels, predictions, target_names=['Authentic', 'Tampered'], zero_division=1)
    print(f"{method.upper()} Classification Report: \n", report)

    accuracy = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, pos_label='Tampered')

    print(f"{method.upper()} Accuracy: {accuracy:.4f}")
    print(f"{method.upper()} Recall (Tampered): {recall:.4f}")

    cm = confusion_matrix(labels, predictions)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Authentic', 'Tampered']).plot()
    plt.title(f'Confusion Matrix - {method.upper()}')
    plt.show()

    return report


def test_classifier(new_dataset_dir, classifier_file_lbp, classifier_file_ltp, classifier_file_fft_eltp, result_file):
    if os.path.exists(result_file):
        print(f"Test results already exist at {result_file}. Skipping re-execution.")
        with open(result_file, 'r', encoding="utf-8") as f:
            print(f.read())
        return

    images, labels = preprocess_images(new_dataset_dir)
    print(f"Number of loaded images: {len(images)}")
    print(f"Number of labels loaded: {len(labels)}")

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Unique labels and their counts:", dict(zip(unique_labels, counts)))

    if len(unique_labels) < 2:
        raise ValueError("Only one class is present in the dataset.")

    report_lbp = evaluate_classifier('lbp', images, labels, classifier_file_lbp)
    report_ltp = evaluate_classifier('ltp', images, labels, classifier_file_ltp)
    report_fft_eltp = evaluate_classifier('fft_eltp', images, labels, classifier_file_fft_eltp)

    with open(result_file, 'w', encoding="utf-8") as f:
        f.write("LBP\n")
        f.write(report_lbp)
        f.write("\n\nLTP\n")
        f.write(report_ltp)
        f.write("\n\nFFT-ELTP\n")
        f.write(report_fft_eltp)

    print(f"Test results saved to {result_file}")


if __name__ == "__main__":
    new_dataset_dir = os.path.abspath('../data/CASIA1.0')
    result_dir = 'results'
    result_file = os.path.join(result_dir, 'test_results.txt')

    classifier_file_lbp = os.path.join(result_dir, 'lbp_classifier.joblib')
    classifier_file_ltp = os.path.join(result_dir, 'ltp_classifier.joblib')
    classifier_file_fft_eltp = os.path.join(result_dir, 'fft_eltp_classifier.joblib')

    test_classifier(
        new_dataset_dir,
        classifier_file_lbp,
        classifier_file_ltp,
        classifier_file_fft_eltp,
        result_file
    )
