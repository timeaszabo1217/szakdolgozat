import matplotlib.pyplot as plt

# Legjobb eredmények
your_accuracy = 0.66
your_recall = 0.4902

# Korábbi kutatásból származó eredmények
previous_accuracy = 0.8862
previous_recall = 0.78

print(f"Current Accuracy: {your_accuracy}, Previous Accuracy: {previous_accuracy}")
print(f"Current Recall: {your_recall}, Previous Recall: {previous_recall}")

if your_accuracy > previous_accuracy:
    print("Current model has better accuracy.")
else:
    print("Previous model has better accuracy.")

if your_recall > previous_recall:
    print("Current model has better recall.")
else:
    print("Previous model has better recall.")


def plot_comparison(your_accuracy, your_recall, previous_accuracy, previous_recall, output_file):
    metrics = ['Accuracy', 'Recall']
    your_scores = [your_accuracy, your_recall]
    previous_scores = [previous_accuracy, previous_recall]

    x = range(len(metrics))

    plt.figure()
    plt.bar(x, your_scores, width=0.4, label='Current Model', color='green', align='center')
    plt.bar(x, previous_scores, width=0.4, label='Previous Research', color='purple', align='edge')
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Comparison of Current Model and Previous Research')
    plt.xticks(ticks=x, labels=metrics)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(output_file)
    plt.close()
    print(f"Comparison plot saved to {output_file}")


output_file = 'results/comparison_plot.png'
plot_comparison(your_accuracy, your_recall, previous_accuracy, previous_recall, output_file)
