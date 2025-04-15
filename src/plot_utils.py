import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from feature_extraction import load_features


def plot_confusion_matrix(cm, method, comp, output_dir):
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Authentic', 'Tampered'])
    cm_plot_file = os.path.join(output_dir, f'{method}_confusion_matrix_{comp}.png')
    cm_display.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {method.upper()} ({comp})')
    plt.savefig(cm_plot_file)
    plt.close()
    print(f"Confusion matrix for {comp} saved to {cm_plot_file}")


def plot_classification_report(report, method, comp, output_dir):
    report_file = os.path.join(output_dir, f'{method}_classification_report_{comp}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Classification report for {method.upper()} ({comp}) saved to {report_file}")


def plot_metrics(accuracy, recall, method, comp, output_dir, test):
    plt.figure()
    metrics = ['Accuracy', 'Recall']
    values = [accuracy * 100, recall * 100]
    plt.bar(metrics, values, color=['green', 'purple'])
    plt.xlabel('Metrics')
    plt.ylabel('Scores (%)')
    plt.title(f'{method.upper()} ({comp}) Classifier Performance Metrics')
    plt.ylim(0, 100)
    if test:
        metrics_plot_file = os.path.join(output_dir, f'test_{method}_{comp}_metrics.png')
    else:
        metrics_plot_file = os.path.join(output_dir, f'{method}_{comp}_metrics.png')
    plt.savefig(metrics_plot_file)
    plt.close()
    print(f"Metrics plot for {method.upper()} ({comp}) saved to {metrics_plot_file}")


def plot_pca_comparison(result_dir, components, methods, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    method_colors = {
        'lbp': 'green',
        'ltp': 'blue',
        'fft_eltp': 'purple'
    }

    for comp in components:
        plt.figure(figsize=(8, 6))
        for method in methods:
            features_file = os.path.join(result_dir, f'{method}_features_labels_{comp}.joblib')
            if not os.path.exists(features_file):
                print(f"Missing: {features_file}")
                continue

            features, labels = load_features(features_file)
            features = StandardScaler().fit_transform(features)

            pca = PCA(n_components=2)
            reduced = pca.fit_transform(features)

            label = method.upper()
            color = method_colors[method]
            plt.scatter(reduced[:, 0], reduced[:, 1], label=label, alpha=0.5, s=15, color=color)

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA Projection - Component: {comp}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pca_{comp}.png'))
        plt.close()
        print(f"PCA plot saved for {comp} to {os.path.join(output_dir, f'pca_{comp}.png')}")


def plot_roc_curve(y_test, y_pred_prob, method, comp, output_dir):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='purple', lw=2, label=f'ROC curve (area = {roc_auc: .2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{method.upper()} - ROC Curve ({comp})')
    plt.legend(loc='lower right')

    roc_curve_file = os.path.join(output_dir, f'{method}_roc_curve_{comp}.png')
    plt.savefig(roc_curve_file)
    plt.close()

    print(f"ROC Curve for {method.upper()} ({comp}) saved to {roc_curve_file}")


def plot_learning_curve(classifier, X_train, y_train, method, comp, output_dir):
    train_sizes, train_scores, test_scores = learning_curve(classifier, X_train, y_train, cv=10, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label='Training score', color='purple')  # type: ignore
    plt.plot(train_sizes, test_scores_mean, label='Cross-validation score', color='green')  # type: ignore
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(f'{method.upper()} - Learning Curve ({comp})')
    plt.legend(loc='best')

    learning_curve_file = os.path.join(output_dir, f'{method}_learning_curve_{comp}.png')
    plt.savefig(learning_curve_file)
    plt.close()

    print(f"Learning Curve for {method.upper()} ({comp}) saved to {learning_curve_file}")


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
