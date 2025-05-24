import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from feature_extraction import load_features
from plot_utils import plot_data_distribution, plot_metrics, plot_roc_curve, plot_confusion_matrix


def train_and_evaluate(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, stratify=labels)

    pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True))])

    param_grid = {
        'svm__C': [1],
        'svm__gamma': [0.3225],
        'svm__kernel': ['rbf'],
        'svm__class_weight': ['balanced']
    }

    grid = GridSearchCV(pipeline, param_grid, cv=10, refit=True, verbose=2, return_train_score=True)
    grid.fit(X_train, y_train)

    classifier = grid.best_estimator_

    print("Best parameters found: ", grid.best_params_)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro')

    return classifier, accuracy, recall, f1, X_train, X_test, y_train, y_test


def save_classifier(classifier, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    joblib.dump(classifier, output_file)
    print(f"Classifier saved to {output_file}")


def save_metrics(accuracy, recall, f1, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        file.write(f"Accuracy: {accuracy * 100: .2f}%\n")
        file.write(f"Recall: {recall * 100: .2f}%\n")
        file.write(f"F1-score: {f1 * 100: .2f}%\n")
    print(f"Metrics saved to {output_file}")


def load_classifier(file_path):
    classifier = joblib.load(file_path)
    print(f"Loaded classifier type: {type(classifier)}")
    return classifier


def process_features(methods, components, results_dir, metrics_dir, plots_dir):
    for method in methods:
        for comp in components:
            features_file = os.path.join(results_dir, f'{method}_features_labels_{comp}.joblib')
            classifier_file = os.path.join(results_dir, f'{method}_classifier_{comp}.joblib')
            metrics_file = os.path.join(metrics_dir, f'{method}_evaluation_metrics_{comp}.txt')
            distribution_plot_file = os.path.join(plots_dir, f'data_distribution.png')
            roc_plot_file = os.path.join(plots_dir, f'{method}_roc_curve_{comp}.png')
            confusion_matrix_file = os.path.join(plots_dir, f'{method}_confusion_matrix_{comp}.png')

            if not os.path.exists(features_file):
                print(f"Missing features file for {method.upper()} ({comp}). Skipping.")
                continue

            print(f"Loading {method.upper()} features ({comp}) from {features_file}")
            features, labels = load_features(features_file)

            if os.path.exists(classifier_file):
                print(f"Loading classifier from {classifier_file}")
                classifier = load_classifier(classifier_file)
                X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, stratify=labels)
            else:
                classifier, accuracy, recall, f1, X_train, X_test, y_train, y_test = train_and_evaluate(features,
                                                                                                        labels)
                save_classifier(classifier, classifier_file)

            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred, zero_division=1)
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f'{method.upper()} ({comp}) Accuracy: {accuracy * 100: .2f}%')
            print(f'{method.upper()} ({comp}) Recall: {recall * 100: .2f}%')
            print(f'{method.upper()} ({comp}) F1-score: {f1 * 100: .2f}%')

            save_metrics(accuracy, recall, f1, metrics_file)
            plot_metrics(accuracy, recall, method, comp, plots_dir, test=False)

            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cm, method, comp, confusion_matrix_file, test=False)

            if len(classifier.classes_) == 2:
                positive_class = max(classifier.classes_)
                positive_index = list(classifier.classes_).index(positive_class)
                y_pred_prob = classifier.predict_proba(X_test)[:, positive_index]
                auc = roc_auc_score(y_test, y_pred_prob)
            else:
                y_pred_prob = classifier.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr', average='macro')
            print(f"ROC AUC manually: {auc: .4f}")
            plot_roc_curve(y_test, y_pred_prob, method, comp, roc_plot_file)

            if not os.path.exists(distribution_plot_file):
                plot_data_distribution(labels, f"Data Distribution", distribution_plot_file)

            result_file = os.path.join(results_dir, 'results.txt')
            with open(result_file, 'a', encoding="utf-8") as file:
                file.write(f"{method.upper()} ({comp}) classification results: \n")
                file.write(f"Number of images: {len(labels)}\n")
                file.write(f"Model type: {classifier}\n")
                file.write(f"Accuracy: {accuracy * 100: .2f}%\n")
                file.write(f"Recall rate: {recall * 100: .2f}%\n")
                file.write(f"F1-score: {f1 * 100: .2f}%\n\n")


if __name__ == "__main__":
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    metrics_dir = os.path.join(results_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    methods = ['lbp', 'ltp', 'fft_eltp']
    components = ['CbCr']  # , 'Cb', 'Cr'

    process_features(methods, components, results_dir, metrics_dir, plots_dir)
