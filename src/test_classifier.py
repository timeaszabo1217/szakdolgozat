import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
from preprocess import preprocess_images
from feature_extraction import extract_features
from train_classifier import load_classifier
from plot_utils import plot_confusion_matrix, plot_classification_report, plot_metrics


def evaluate_classifier(images, labels, methods, components, classifier_dir, plots_dir):
    all_reports = []

    for method in methods:
        for comp in components:
            print(f"Processing {method.upper()} with {comp} component")

            features = extract_features(images, labels, [method], [comp], output_file_base=None)
            print(f"Number of {method.upper()} features extracted for {comp}: {features.shape}")

            if features.size == 0:
                raise ValueError(f"Extracted {method.upper()} features for {comp} are empty.")

            classifier_file = os.path.join(classifier_dir, f'{method}_classifier_{comp}.joblib')
            classifier = load_classifier(classifier_file)
            print(f"{method.upper()} Classifier parameters: ", classifier.get_params())

            predictions = classifier.predict(features)

            report = classification_report(labels, predictions, target_names=['Authentic', 'Tampered'], zero_division=1)
            print(f"{method.upper()} Classification Report for {comp}: \n", report)

            accuracy = accuracy_score(labels, predictions)
            recall = recall_score(labels, predictions, pos_label=1)

            print(f'{method.upper()} ({comp}) Accuracy: {accuracy * 100: .2f}%')
            print(f'{method.upper()} ({comp}) Recall: {recall * 100: .2f}%')

            cm = confusion_matrix(labels, predictions)

            plot_confusion_matrix(cm, method, comp, plots_dir, test=True)
            plot_classification_report(report, method, comp, plots_dir)
            plot_metrics(accuracy, recall, method, comp, plots_dir, test=True)

            all_reports.append(f"{method.upper()} with {comp}: \n{report}")

    return "\n\n".join(all_reports)


def test_classifier(dataset_dir, methods, components, result_file, classifier_dir, plots_dir):
    if os.path.exists(result_file):
        print(f"Test results already exist at {result_file}. Skipping testing.")
        with open(result_file, 'r', encoding="utf-8") as f:
            print(f.read())
        return

    images, labels = preprocess_images(dataset_dir)
    print(f"Number of loaded images: {len(images)}")
    print(f"Number of labels loaded: {len(labels)}")

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Unique labels and their counts:", dict(zip(unique_labels, counts)))

    if len(unique_labels) < 2:
        raise ValueError("Warning: Only one class is present in the dataset.")

    all_reports = evaluate_classifier(images, labels, methods, components, classifier_dir, plots_dir)

    with open(result_file, 'w', encoding="utf-8") as f:
        f.write(all_reports)

    print(f"Test results saved to {result_file}")


if __name__ == "__main__":
    test_dir = os.path.abspath('../data/CASIA2.0_revised')
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    result_file = os.path.join(results_dir, 'test_results.txt')

    methods = ['lbp', 'ltp', 'fft_eltp']
    components = ['CbCr', 'Cb', 'Cr']

    test_classifier(test_dir, methods, components, result_file, results_dir, plots_dir)
