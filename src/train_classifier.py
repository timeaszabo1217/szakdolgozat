import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score
from feature_extraction import load_features, extract_features_lbp_ltp, extract_features_fft_eltp, save_features
from preprocess import preprocess_images


def train_and_evaluate(features, labels):
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, stratify=labels)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.5, 0.3225, 0.3, 0.1],
        'kernel': ['rbf']
    }
    grid = GridSearchCV(svm.SVC(), param_grid, cv=10, refit=True, verbose=2)
    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best estimator: {grid.best_estimator_}")

    classifier = grid.best_estimator_
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, zero_division=1)
    return classifier, accuracy, recall


def save_classifier(classifier, output_file):
    with open(output_file, 'wb') as file:
        pickle.dump(classifier, file)
    print(f"Classifier saved to {output_file}")


def save_metrics(accuracy, recall, output_file):
    with open(output_file, 'w') as file:
        file.write(f'Accuracy: {accuracy * 100: .2f}%\n')
        file.write(f'Recall: {recall * 100: .2f}%\n')
    print(f"Metrics saved to {output_file}")


def load_classifier(file_path):
    with open(file_path, 'rb') as file:
        classifier = pickle.load(file)
    print(f"Loaded classifier type: {type(classifier)}")
    return classifier


def plot_metrics(accuracy, recall, output_file):
    plt.figure()
    metrics = ['Accuracy', 'Recall']
    values = [accuracy * 100, recall * 100]
    plt.bar(metrics, values, color=['green', 'purple'])
    plt.xlabel('Metrics')
    plt.ylabel('Scores (%)')
    plt.title('Classifier Performance Metrics')
    plt.ylim(0, 100)
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")


def plot_data_distribution(labels, title):
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Unique labels: {unique}, Counts: {counts}")
    plt.figure()
    plt.bar(unique, counts, color=['green', 'purple'])
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.xticks(unique, ['Authentic', 'Tampered'][:len(unique)])
    plt.show()
    return unique, counts


if __name__ == "__main__":
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)
    features_file_lbp_ltp = os.path.join(result_dir, 'lbp_ltp_features_labels.npz')
    features_file_fft_eltp = os.path.join(result_dir, 'fft_eltp_features_labels.npz')
    classifier_file_lbp_ltp = os.path.join(result_dir, 'classifier_lbp_ltp.pkl')
    classifier_file_fft_eltp = os.path.join(result_dir, 'classifier_fft_eltp.pkl')
    metrics_file = os.path.join(result_dir, 'evaluation_metrics.txt')
    plot_file = os.path.join(result_dir, 'metrics_plot.png')

    revised_dir = os.path.abspath('../data/CASIA2.0_revised')

    # LBP és LTP jellemzők kezelése
    if os.path.exists(features_file_lbp_ltp):
        print(f"Loading LBP and LTP features from {features_file_lbp_ltp}")
        features_lbp_ltp, labels = load_features(features_file_lbp_ltp)
    else:
        images, labels = preprocess_images(revised_dir)
        features_lbp_ltp = extract_features_lbp_ltp(images)
        save_features(features_lbp_ltp, labels, features_file_lbp_ltp)

    # FFT és ELTP jellemzők kezelése
    if os.path.exists(features_file_fft_eltp):
        print(f"Loading FFT-ELTP features from {features_file_fft_eltp}")
        features_fft_eltp, labels = load_features(features_file_fft_eltp)
    else:
        images, labels = preprocess_images(revised_dir)
        features_fft_eltp = extract_features_fft_eltp(images)
        save_features(features_fft_eltp, labels, features_file_fft_eltp)

    # LBP-LTP osztályozó betanítása, értékelése
    if os.path.exists(classifier_file_lbp_ltp):
        print(f"Loading classifier from {classifier_file_lbp_ltp}")
        classifier_lbp_ltp = load_classifier(classifier_file_lbp_ltp)
    else:
        classifier_lbp_ltp, accuracy_lbp_ltp, recall_lbp_ltp = train_and_evaluate(features_lbp_ltp, labels)
        print(f'LBP-LTP Accuracy: {accuracy_lbp_ltp * 100: .2f}%')
        print(f'LBP-LTP Recall: {recall_lbp_ltp * 100: .2f}%')
        save_classifier(classifier_lbp_ltp, classifier_file_lbp_ltp)

        save_metrics(accuracy_lbp_ltp, recall_lbp_ltp, metrics_file.replace('.txt', '_lbp_ltp.txt'))
        plot_metrics(accuracy_lbp_ltp, recall_lbp_ltp, plot_file.replace('.png', '_lbp_ltp.png'))

        unique_labels, counts = plot_data_distribution(labels, "Data Distribution for LBP_LTP")

        result_file = os.path.join('results', 'results.txt')
        with open(result_file, 'a', encoding="utf-8") as file:
            file.write("LBP-LTP classification results:\n")
            file.write(f"Number of images: {len(labels)}\n")
            file.write(f"Best parameters: {classifier_lbp_ltp.get_params()}\n")
            file.write(f"Model type: {classifier_lbp_ltp}\n")
            file.write(f"Number of images classified as authentic: {counts[0]}\n")
            file.write(f"Number of images classified as fake: {counts[1]}\n\n")
            file.write(f"Accuracy: {accuracy_lbp_ltp * 100: .2f}%\n")
            file.write(f"Recall rate: {recall_lbp_ltp * 100: .2f}%\n\n")

    # FFT-ELTP osztályozó betanítása, értékelése
    if os.path.exists(classifier_file_fft_eltp):
        print(f"Loading classifier from {classifier_file_fft_eltp}")
        classifier_fft_eltp = load_classifier(classifier_file_fft_eltp)
    else:
        classifier_fft_eltp, accuracy_fft_eltp, recall_fft_eltp = train_and_evaluate(features_fft_eltp, labels)
        print(f'FFT-ELTP Accuracy: {accuracy_fft_eltp * 100: .2f}%')
        print(f'FFT-ELTP Recall: {recall_fft_eltp * 100: .2f}%')
        save_classifier(classifier_fft_eltp, classifier_file_fft_eltp)

        save_metrics(accuracy_fft_eltp, recall_fft_eltp, metrics_file.replace('.txt', '_fft_eltp.txt'))
        plot_metrics(accuracy_fft_eltp, recall_fft_eltp, plot_file.replace('.png', '_fft_eltp.png'))

        unique_labels, counts = plot_data_distribution(labels, "Data Distribution for FFT-ELTP")

        result_file = os.path.join('results', 'results.txt')
        with open(result_file, 'a', encoding="utf-8") as file:
            file.write("FFT-ELTP classification results:\n")
            file.write(f"Number of images: {len(labels)}\n")
            file.write(f"Best parameters: {classifier_fft_eltp.get_params()}\n")
            file.write(f"Model type: {classifier_fft_eltp}\n")
            file.write(f"Number of images classified as authentic: {counts[0]}\n")
            file.write(f"Number of images classified as fake: {counts[1]}\n\n")
            file.write(f"Accuracy: {accuracy_fft_eltp * 100: .2f}%\n")
            file.write(f"Recall rate: {recall_fft_eltp * 100: .2f}%\n\n")