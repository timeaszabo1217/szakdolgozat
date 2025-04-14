import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, accuracy_score, recall_score
from preprocess import preprocess_images
from feature_extraction import extract_features
from train_classifier import load_classifier


def evaluate_classifier(method, images, labels, classifier_file, result_dir, components):
    reports = []
    for component in components:
        print(f"Processing {method.upper()} with {component} component")

        features = extract_features(images, labels, method=method, output_file_base=None)
        print(f"Number of {method.upper()} features extracted for {component}: {features.shape}")

        if features.size == 0:
            raise ValueError(f"Extracted {method.upper()} features for {component} are empty.")

        classifier = load_classifier(classifier_file)
        print(f"{method.upper()} Classifier parameters: ", classifier.get_params())

        predictions = classifier.predict(features)

        report = classification_report(labels, predictions, target_names=['Authentic', 'Tampered'], zero_division=1)
        print(f"{method.upper()} Classification Report for {component}: \n", report)

        accuracy = accuracy_score(labels, predictions)
        recall = recall_score(labels, predictions, pos_label=1)

        print(f'{method.upper()} Accuracy for {component}: {accuracy * 100: .2f}%')
        print(f'{method.upper()} Recall for {component}: {recall * 100: .2f}%')

        cm = confusion_matrix(labels, predictions)

        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Authentic', 'Tampered'])
        cm_plot_file = os.path.join(result_dir, f'confusion_matrix_{method}_{component}.png')
        cm_display.plot(cmap='Blues')
        plt.title(f'Confusion Matrix - {method.upper()} ({component})')
        plt.savefig(cm_plot_file)
        plt.close()

        print(f"Confusion matrix for {component} saved to {cm_plot_file}")

        reports.append(f"Report for {component}:\n{report}")

    return "\n\n".join(reports)


def test_classifier(new_dataset_dir, classifier_file_lbp, classifier_file_ltp, classifier_file_fft_eltp, result_file,
                    result_dir):
    if os.path.exists(result_file):
        print(f"Test results already exist at {result_file}. Skipping testing.")
        with open(result_file, 'r', encoding="utf-8") as f:
            print(f.read())
        return

    images, labels = preprocess_images(new_dataset_dir)
    print(f"Number of loaded images: {len(images)}")
    print(f"Number of labels loaded: {len(labels)}")

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Unique labels and their counts:", dict(zip(unique_labels, counts)))

    if len(unique_labels) < 2:
        raise ValueError("Warning: Only one class is present in the dataset.")

    components = ['Cb', 'Cr', 'CbCr']
    report_lbp = evaluate_classifier('lbp', images, labels, classifier_file_lbp, result_dir, components)
    report_ltp = evaluate_classifier('ltp', images, labels, classifier_file_ltp, result_dir, components)
    report_fft_eltp = evaluate_classifier('fft_eltp', images, labels, classifier_file_fft_eltp, result_dir, components)

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
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, 'test_results.txt')

    classifier_file_lbp = os.path.join(result_dir, 'lbp_classifier.joblib')
    classifier_file_ltp = os.path.join(result_dir, 'ltp_classifier.joblib')
    classifier_file_fft_eltp = os.path.join(result_dir, 'fft_eltp_classifier.joblib')

    test_classifier(
        new_dataset_dir,
        classifier_file_lbp,
        classifier_file_ltp,
        classifier_file_fft_eltp,
        result_file,
        result_dir
    )
