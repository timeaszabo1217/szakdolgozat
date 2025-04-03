import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from feature_extraction import load_features, extract_features, save_features
from preprocess import preprocess_images


def train_and_evaluate(features, labels):
    if features.ndim == 1:
        features = features.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, stratify=labels)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC())
    ])

    param_grid = {
        'svm__C': [0.1, 1, 10, 100, 1000],
        'svm__gamma': [0.3225],
        'svm__kernel': ['rbf'],
        'svm__class_weight': ['balanced']
    }

    grid = GridSearchCV(pipeline, param_grid, cv=10, refit=True, verbose=2)
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
        file.write(f"Accuracy: {accuracy * 100: .2f}%\n")
        file.write(f"Recall: {recall * 100: .2f}%\n")
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


def process_features(revised_dir, result_dir, methods, batch_size=50):
    for method in methods:
        features_file = os.path.join(result_dir, f'{method}_features_labels.npz')
        classifier_file = os.path.join(result_dir, f'{method}_classifier.pkl')
        metrics_file = os.path.join(result_dir, f'{method}_evaluation_metrics.txt')
        plot_file = os.path.join(result_dir, f'{method}_metrics_plot.png')

        if os.path.exists(features_file):
            print(f"Loading {method.upper()} features from {features_file}")
            features, labels = load_features(features_file)
        else:
            images, labels = preprocess_images(revised_dir)
            features = []

            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                print(f"Processing batch {i // batch_size + 1} of {len(images) // batch_size + 1}...")
                batch_features = extract_features(batch_images, method)
                features.append(batch_features)

            features = np.concatenate(features, axis=0)
            save_features(features, labels, features_file)

        if os.path.exists(classifier_file):
            print(f"Loading classifier from {classifier_file}")
            classifier = load_classifier(classifier_file)

            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1,
                                                                stratify=labels)

            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            print(f'{method.upper()} Accuracy: {accuracy * 100: .2f}%')
            print(f'{method.upper()} Recall: {recall * 100: .2f}%')
        else:
            classifier, accuracy, recall = train_and_evaluate(features, labels)
            print(f'{method.upper()} Accuracy: {accuracy * 100: .2f}%')
            print(f'{method.upper()} Recall: {recall * 100: .2f}%')
            save_classifier(classifier, classifier_file)

        save_metrics(accuracy, recall, metrics_file)
        plot_metrics(accuracy, recall, plot_file)

        unique_labels, counts = plot_data_distribution(labels, f"Data Distribution for {method.upper()}")

        result_file = os.path.join(result_dir, 'results.txt')
        with open(result_file, 'a', encoding="utf-8") as file:
            file.write(f"{method.upper()} classification results: \n")
            file.write(f"Number of images: {len(labels)}\n")
            file.write(f"Best parameters: {classifier.get_params()}\n")
            file.write(f"Model type: {classifier}\n")
            file.write(f"Number of images classified as authentic: {counts[0]}\n")
            file.write(f"Number of images classified as fake: {counts[1]}\n\n")
            file.write(f"Accuracy: {accuracy * 100: .2f}%\n")
            file.write(f"Recall rate: {recall * 100: .2f}%\n\n")


if __name__ == "__main__":
    revised_dir = os.path.abspath('../data/CASIA2.0_revised')
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)

    methods = ['lbp', 'ltp', 'fft_eltp']

    process_features(revised_dir, result_dir, methods, batch_size=50)