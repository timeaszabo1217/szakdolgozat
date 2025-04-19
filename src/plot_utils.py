import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay


def plot_confusion_matrix(cm, method, comp, output_dir):
    display_labels = ['Authentic', 'Tampered']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(f'Confusion Matrix - {method.upper()} - {comp}')

    filename = f'test_{method}_confusion_matrix_{comp}.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def plot_classification_report(report_text, method, comp, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.01, 0.05, report_text, {'fontsize': 10}, fontproperties='monospace')
    ax.axis('off')
    plt.title(f'Classification Report - {method.upper()} - {comp}')
    filename = f'test_{method}_classification_report_{comp}.png'
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_metrics(accuracy, recall, method, comp, output_dir, test=False):
    fig, ax = plt.subplots(figsize=(4, 4))
    metrics = {'Accuracy': accuracy, 'Recall': recall}
    ax.bar(metrics.keys(), metrics.values(), color=['purple', 'green'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')

    prefix = 'test_' if test else ''
    filename = f'{prefix}{method}_metrics_plot_{comp}.png'

    output_path = os.path.join(output_dir, filename)

    plt.title(f'{method.upper()} Metrics - {comp}')
    plt.savefig(output_path)
    plt.close()
    print(f"Metrics plot saved to: {output_path}")


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
    plt.bar(unique, counts, color=['purple', 'green'])
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.xticks(unique, ['Authentic', 'Tampered'][:len(unique)])
    plt.savefig(output_file)
    plt.close()
    return unique, counts
