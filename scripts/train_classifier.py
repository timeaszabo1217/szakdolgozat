import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score


def train_and_evaluate(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)
    classifier = svm.SVC(kernel='rbf', gamma=0.3225)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return classifier, accuracy, recall


def save_classifier(classifier, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"Classifier saved to {output_file}")


def save_metrics(accuracy, recall, output_file):
    with open(output_file, 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Recall: {recall}\n')
    print(f"Metrics saved to {output_file}")


def load_classifier(file_path):
    with open(file_path, 'rb') as f:
        classifier = pickle.load(f)
    print(f"Loaded classifier type: {type(classifier)}")
    return classifier


def plot_metrics(accuracy, recall, output_file):
    plt.figure()
    metrics = ['Accuracy', 'Recall']
    values = [accuracy, recall]
    plt.bar(metrics, values, color=['blue', 'orange'])
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Classifier Performance Metrics')
    plt.ylim(0, 1)
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")


# Adatok megoszlásának vizualizálása
def plot_data_distribution(labels, title):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure()
    plt.bar(unique, counts, color=['blue', 'orange'])
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.xticks(unique, ['Authentic', 'Tampered'])
    plt.show()


if __name__ == "__main__":
    from feature_extraction import load_features, save_features

    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)
    features_file = os.path.join(result_dir, 'features_labels.npz')
    classifier_file = os.path.join(result_dir, 'classifier_model.pkl')
    metrics_file = os.path.join(result_dir, 'evaluation_metrics.txt')
    plot_file = os.path.join(result_dir, 'metrics_plot.png')

    if os.path.exists(features_file):
        print(f"Loading features from {features_file}")
        features, labels = load_features(features_file)
    else:
        from preprocess import preprocess_images
        from feature_extraction import extract_features

        revised_dir = os.path.abspath('../data/CASIA2.0_revised')
        images, labels = preprocess_images(revised_dir)
        features = extract_features(images)
        save_features(features, labels, features_file)

    if os.path.exists(classifier_file):
        print(f"Loading classifier from {classifier_file}")
        classifier = load_classifier(classifier_file)
        with open(metrics_file, 'r') as f:
            metrics = f.read()
        print(f"Loaded metrics: \n{metrics}")
    else:
        classifier, accuracy, recall = train_and_evaluate(features, labels)
        print(f'Accuracy: {accuracy}')
        print(f'Recall: {recall}')
        save_classifier(classifier, classifier_file)
        save_metrics(accuracy, recall, metrics_file)
        plot_metrics(accuracy, recall, plot_file)

    # Adatok megoszlásának vizualizálása
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)
    plot_data_distribution(y_train, 'Training Set Distribution')
    plot_data_distribution(y_test, 'Test Set Distribution')
