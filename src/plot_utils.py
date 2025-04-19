import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay


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


def plot_metrics(accuracy, recall, method, comp, output_file, test):
    if test:
        directory = os.path.dirname(output_file)
        filename = os.path.basename(output_file)
        output_file = os.path.join(directory, f'test_{filename}')

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.figure()
    metrics = ['Accuracy', 'Recall']
    values = [accuracy * 100, recall * 100]
    plt.bar(metrics, values, color=['green', 'purple'])
    plt.xlabel('Metrics')
    plt.ylabel('Scores (%)')
    plt.title(f'{method.upper()} ({comp}) Classifier Performance Metrics')
    plt.ylim(0, 100)
    plt.savefig(output_file)
    plt.close()

    print(f"Metrics plot for {method.upper()} ({comp}) saved to {output_file}")


def plot_roc_curve(y_test, y_pred_prob, method, comp, output_file):
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
    plt.savefig(output_file)
    plt.close()

    print(f"ROC Curve for {method.upper()} ({comp}) saved to {output_file}")


def plot_data_distribution(labels, title, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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
