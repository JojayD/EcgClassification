import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def visualize_ecg_with_symbols(ecg_signal, predicted_class, timestamp, symbol_mapping):
    """
    Overlay symbolic representation on ECG signal.

    Parameters:
    - ecg_signal (list of tuples): [(x1, y1), (x2, y2), ...]
    - predicted_class (int): Predicted class index.
    - timestamp (float): Time at which to place the symbol.
    - symbol_mapping (dict): Mapping from class index to symbols.
    """
    x_values = [point[0] for point in ecg_signal]
    y_values = [point[1] for point in ecg_signal]
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, y_values, label='ECG Signal')
    plt.title('ECG Signal with Symbolic Overlay')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Overla
    plt.text(timestamp, max(y_values), f'Symbol: {predicted_class}', fontsize=12, color='red')

    plt.legend()
    plt.show()


def visualize_embeddings(embeddings, labels, class_mapping, title='ECG Embeddings Visualization'):
    """
    Visualize high-dimensional embeddings using t-SNE.

    Parameters:
    - embeddings (numpy.ndarray): Array of embeddings.
    - labels (list or numpy.ndarray): Corresponding class labels.
    - class_mapping (dict): Mapping from class index to class names.
    - title (str): Title of the plot.
    """
    # Ensure labels are of type int
    labels = [int(label) for label in labels]
    print(labels)
    n_samples = embeddings.shape[0]
    perplexity = min(30, n_samples - 1)  # As a rule of thumb, perplexity should be less than n_samples

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1],
                    hue=labels, palette='viridis', alpha=0.7,
                    legend='full')
    reverse_class_mapping = {v: k for k, v in class_mapping.items()}
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title='Classes', labels=[reverse_class_mapping[i] for i in set(labels)])
    plt.grid(True)
    plt.show()