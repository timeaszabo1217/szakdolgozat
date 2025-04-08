import matplotlib.pyplot as plt

my_accuracy = 0.6839

previous_accuracy = 0.8862

print(f"Current Accuracy: {my_accuracy}, Previous Accuracy: {previous_accuracy}")

if my_accuracy > previous_accuracy:
    print("Current model has better accuracy.")
else:
    print("Previous model has better accuracy.")


def plot_comparison(your_accuracy, previous_accuracy, output_file):
    models = ['Current Model', 'Previous Research']
    accuracies = [your_accuracy, previous_accuracy]

    x = range(len(models))

    plt.figure()
    plt.bar(x, accuracies, width=0.4, label='Model Accuracy', color='green', align='center')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Accuracy: Current Model vs. Previous Research')
    plt.xticks(ticks=x, labels=models)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(output_file)
    plt.close()
    print(f"Comparison plot saved to {output_file}")


output_file = 'results/comparison_plot.png'
plot_comparison(my_accuracy, previous_accuracy, output_file)
