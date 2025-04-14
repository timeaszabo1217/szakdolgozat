import os
import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, recall_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from feature_extraction import load_features


def train_and_evaluate(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, stratify=labels)

    variance = np.var(X_train, axis=0)
    gamma = 1 / (X_train.shape[1] * np.mean(variance))

    print(f"Calculated gamma: {gamma}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True))
    ])

    param_grid = {
        'svm__C': [0.001, 0.1, 1, 10, 100, 1000],
        'svm__gamma': [gamma],
        'svm__kernel': ['rbf'],
        'svm__class_weight': ['balanced']
    }

    grid = GridSearchCV(pipeline, param_grid, cv=10, refit=True, verbose=3, return_train_score=True)
    grid.fit(X_train, y_train)

    classifier = grid.best_estimator_

    print("Best parameters found: ", grid.best_params_)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, zero_division=1)

    return classifier, accuracy, recall, X_train, X_test, y_train, y_test


def plot_roc_curve(y_test, y_pred_prob, output_file):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='purple', lw=2, label=f'ROC curve (area = {roc_auc: .2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(output_file)
    plt.close()

    print(f"ROC Curve saved to {output_file}")


def plot_learning_curve(classifier, X_train, y_train, output_file):
    train_sizes, train_scores, test_scores = learning_curve(classifier, X_train, y_train, cv=10, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label='Training score', color='purple')  # type: ignore
    plt.plot(train_sizes, test_scores_mean, label='Cross-validation score', color='green')  # type: ignore
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.savefig(output_file)
    plt.close()

    print(f"Learning Curve saved to {output_file}")


def save_classifier(classifier, output_file):
    joblib.dump(classifier, output_file)
    print(f"Classifier saved to {output_file}")


def save_metrics(accuracy, recall, output_file):
    with open(output_file, 'w') as file:
        file.write(f"Accuracy: {accuracy * 100: .2f}%\n")
        file.write(f"Recall: {recall * 100: .2f}%\n")
    print(f"Metrics saved to {output_file}")


def load_classifier(file_path):
    classifier = joblib.load(file_path)
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


def plot_data_distribution(labels, title, output_file):
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Unique labels: {unique}, Counts: {counts}")
    plt.figure()
    plt.bar(unique, counts, color=['green', 'purple'])
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.xticks(unique, ['Authentic', 'Tampered'][:len(unique)])
    plt.savefig(output_file)
    plt.close()
    return unique, counts


def process_features(result_dir, methods, components):
    for method in methods:
        for comp in components:
            comp_suffix = f"_{comp}"
            features_file = os.path.join(result_dir, f'{method}_features_labels{comp_suffix}.joblib')
            classifier_file = os.path.join(result_dir, f'{method}_classifier{comp_suffix}.joblib')
            evaluation_metrics_text_file = os.path.join(metrics_dir, f'{method}_evaluation_metrics{comp_suffix}.txt')
            distribution_plot_file = os.path.join(plots_dir, 'data_distribution.png')
            metrics_plot_file = os.path.join(plots_dir, f'{method}_metrics_plot{comp_suffix}.png')
            roc_plot_file = os.path.join(plots_dir, f'{method}_roc_curve{comp_suffix}.png')
            learning_curve_plot_file = os.path.join(plots_dir, f'{method}_learning_curve{comp_suffix}.png')

            if os.path.exists(features_file):
                print(f"Loading {method.upper()} features ({comp}) from {features_file}")
                features, labels = load_features(features_file)
            else:
                print(f"Missing features file for {method.upper()} ({comp}). Skipping.")
                continue

            if os.path.exists(classifier_file):
                print(f"Loading classifier from {classifier_file}")
                classifier = load_classifier(classifier_file)
                X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, stratify=labels)
                y_pred = classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
            else:
                classifier, accuracy, recall, X_train, X_test, y_train, y_test = train_and_evaluate(features, labels)
                save_classifier(classifier, classifier_file)

            print(f'{method.upper()} ({comp}) Accuracy: {accuracy * 100: .2f}%')
            print(f'{method.upper()} ({comp}) Recall: {recall * 100: .2f}%')

            save_metrics(accuracy, recall, evaluation_metrics_text_file)
            plot_metrics(accuracy, recall, metrics_plot_file)

            y_pred_prob = classifier.predict_proba(X_test)[:, 1]
            plot_roc_curve(y_test, y_pred_prob, roc_plot_file)

            plot_learning_curve(classifier, X_train, y_train, learning_curve_plot_file)

            unique_labels, counts = plot_data_distribution(labels, f"Data Distribution for {method.upper()} ({comp})", distribution_plot_file)

            result_file = os.path.join(result_dir, 'results.txt')
            with open(result_file, 'a', encoding="utf-8") as file:
                file.write(f"{method.upper()} ({comp}) classification results: \n")
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
    metrics_dir = os.path.join(result_dir, 'evaluation_metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    plots_dir = os.path.join(result_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    methods = ['lbp', 'ltp', 'fft_eltp']
    components = ['CbCr', 'Cb', 'Cr']

    process_features(result_dir, methods, components)
