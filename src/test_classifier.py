import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
from preprocess import preprocess_images
from feature_extraction import extract_features
from train_classifier import load_classifier
from plot_utils import plot_confusion_matrix, plot_classification_report, plot_metrics


def evaluate_classifier(method, component, images, labels, classifier_file, output_dir):
    print(f"Processing {method.upper()} with {component} component")

    features = extract_features(images, labels, method, [component], output_file_base=None)
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

    plot_confusion_matrix(cm, method, component, output_dir)
    plot_classification_report(report, method, component, output_dir)
    plot_metrics(accuracy, recall, method, component, output_dir, test=True)

    return f"Report for {component}:\n{report}"


def test_classifier(dataset_dir, result_dir, methods, components, output_file):
    if os.path.exists(output_file):
        print(f"Test results already exist at {output_file}. Skipping testing.")
        with open(output_file, 'r', encoding="utf-8") as f:
            print(f.read())
        return

    images, labels = preprocess_images(dataset_dir)
    print(f"Number of loaded images: {len(images)}")
    print(f"Number of labels loaded: {len(labels)}")

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Unique labels and their counts:", dict(zip(unique_labels, counts)))

    if len(unique_labels) < 2:
        raise ValueError("Warning: Only one class is present in the dataset.")

    all_reports = []

    for method in methods:
        method_reports = [f"{method.upper()}"]
        for comp in components:
            classifier_file = os.path.join(result_dir, f"{method}_classifier_{comp}.joblib")
            if not os.path.exists(classifier_file):
                print(f"Classifier file not found for {method.upper()} - {comp}, skipping.")
                continue

            report = evaluate_classifier(method, comp, images, labels, classifier_file, result_dir)
            method_reports.append(report)
        all_reports.append("\n\n".join(method_reports))

    with open(output_file, 'w', encoding="utf-8") as f:
        f.write("\n\n".join(all_reports))

    print(f"Test results saved to {output_file}")


if __name__ == "__main__":
    test_dir = os.path.abspath('../data/CASIA1.0')
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)
    plots_dir = os.path.join(result_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    result_file = os.path.join(result_dir, 'test_results.txt')

    methods = ['lbp', 'ltp', 'fft_eltp']
    components = ['Cb', 'Cr', 'CbCr']

    test_classifier(test_dir, result_dir, methods, components, result_file)
