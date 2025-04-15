import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
from preprocess import preprocess_images
from feature_extraction import extract_features
from train_classifier import load_classifier
from plot_utils import plot_confusion_matrix, plot_classification_report, plot_metrics


def evaluate_classifier(method, images, labels, classifier_file, result_dir, components):
    reports = []
    for comp in components:
        print(f"Processing {method.upper()} with {comp} component")

        features = extract_features(images, labels, method, [comp], output_file_base=None)
        print(f"Number of {method.upper()} features extracted for {comp}: {features.shape}")

        if features.size == 0:
            raise ValueError(f"Extracted {method.upper()} features for {comp} are empty.")

        classifier = load_classifier(classifier_file)
        print(f"{method.upper()} Classifier parameters: ", classifier.get_params())

        predictions = classifier.predict(features)

        report = classification_report(labels, predictions, target_names=['Authentic', 'Tampered'], zero_division=1)
        print(f"{method.upper()} Classification Report for {comp}: \n", report)

        accuracy = accuracy_score(labels, predictions)
        recall = recall_score(labels, predictions, pos_label=1)

        print(f'{method.upper()} Accuracy for {comp}: {accuracy * 100: .2f}%')
        print(f'{method.upper()} Recall for {comp}: {recall * 100: .2f}%')

        cm = confusion_matrix(labels, predictions)

        plot_confusion_matrix(cm, method, comp, result_dir)
        plot_classification_report(report, method, comp, result_dir)
        plot_metrics(accuracy, recall, method, comp, result_dir, test=True)

        reports.append(f"Report for {comp}: \n{report}")

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
