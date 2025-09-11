import json
import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

data_root = pathlib.Path(__file__).parent.parent.joinpath("data")
label_embeddings_path = data_root.joinpath("label_embeddings.json")


# --- Step 0: Prepare Sample Data ---
# This section creates a sample `label_embeddings` object for demonstration.
# Replace this with your actual data.
def loads_sample_data() -> typing.List[typing.Tuple[str, typing.List[float]]]:
    """Creates a sample dataset with 4 distinct clusters."""
    embeddings: typing.List[typing.Tuple[str, typing.List[float]]] = json.loads(
        label_embeddings_path.read_text()
    )
    return embeddings


# --- Step 1: Data Preparation ---
def prepare_data(
    label_embeddings: typing.List[typing.Tuple[str, typing.List[float]]],
) -> typing.Tuple[typing.List[str], np.ndarray]:
    """
    Separates labels and embeddings from the input data structure.
    """
    labels = [item[0] for item in label_embeddings]
    embeddings_np = np.array([item[1] for item in label_embeddings])
    return labels, embeddings_np


# --- Step 2: Dimensionality Reduction with t-SNE ---
def reduce_dimensions_tsne(embeddings: np.ndarray) -> np.ndarray:
    """
    Reduces the dimensionality of embeddings from 512D to 2D using t-SNE.
    """
    # For high-dimensional data, it's common to first reduce to an intermediate
    # dimension (e.g., 50) with PCA before running t-SNE for better results and speed.
    # However, for simplicity here, we run t-SNE directly.
    tsne = TSNE(
        n_components=2,
        perplexity=min(5, embeddings.shape[0])
        - 1,  # Typical values are between 5 and 50.
        random_state=42,
        init="pca",
        learning_rate="auto",
        max_iter=1000,  # type: ignore
    )
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d


# --- Step 3: Clustering with DBSCAN ---
def perform_dbscan_clustering(embeddings_2d: np.ndarray) -> np.ndarray:
    """
    Performs DBSCAN clustering on the 2D embeddings.
    Note: `eps` and `min_samples` may need tuning for your specific data.
    """
    # The choice of `eps` is crucial. A good starting point can be found by
    # analyzing the distance to the k-th nearest neighbor (often k=min_samples).
    dbscan = DBSCAN(eps=2.5, min_samples=2)  # Larger eps for easier clustering
    cluster_labels = dbscan.fit_predict(embeddings_2d)
    return cluster_labels


# --- Step 4: Python Visualization ---
def plot_clusters(
    embeddings_2d: np.ndarray,
    labels: typing.List[str],
    cluster_labels: np.ndarray,
    output_filename: pathlib.Path | str = data_root.joinpath(
        "cluster_visualization.png"
    ),
):
    """
    Creates and saves a 2D scatter plot of the clustered embeddings.
    """
    unique_clusters = set(cluster_labels)
    # Use a color map that provides distinct colors for clusters.
    # The -1 cluster from DBSCAN represents noise and will be plotted in black.
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))  # type: ignore

    plt.figure(figsize=(16, 12))

    for cluster_id, color in zip(unique_clusters, colors):
        if cluster_id == -1:
            # Color noise points black.
            point_color = "black"
            marker = "x"
            label = "Noise"
        else:
            point_color = color
            marker = "o"
            label = f"Cluster {cluster_id}"

        # Find all points belonging to the current cluster
        class_member_mask = cluster_labels == cluster_id
        xy = embeddings_2d[class_member_mask]

        plt.scatter(
            xy[:, 0],
            xy[:, 1],
            s=50,
            c=[point_color],  # `c` expects a list of colors
            marker=marker,
            label=label,
            alpha=0.8,
            edgecolors="k",
        )

    # Optional: Add text labels to each point.
    # This can be cluttered if you have many points.
    for i, label_text in enumerate(labels):
        plt.annotate(
            label_text,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=8,
            alpha=0.7,
        )

    plt.title("t-SNE Projection with DBSCAN Clustering")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(output_filename)
    print(f"Plot saved to '{output_filename}'")


# --- Main Execution ---
if __name__ == "__main__":
    # 0. Generate or load your data
    label_embeddings = loads_sample_data()

    # 1. Separate labels and vectors
    print("Step 1: Preparing data...")
    labels, embeddings_512d = prepare_data(label_embeddings)

    # 2. Reduce dimensions
    print("Step 2: Reducing dimensions with t-SNE... (this may take a moment)")
    embeddings_2d = reduce_dimensions_tsne(embeddings_512d)

    # 3. Perform clustering
    print("Step 3: Performing DBSCAN clustering...")
    cluster_ids = perform_dbscan_clustering(embeddings_2d)

    # 4. Create and save the plot
    print("Step 4: Generating and saving the plot...")
    plot_clusters(embeddings_2d, labels, cluster_ids)

    print("\nProcess finished.")
    num_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    num_noise = np.sum(cluster_ids == -1)
    print(f"Found {num_clusters} clusters and {num_noise} noise points.")
