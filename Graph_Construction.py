import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import is_undirected, contains_self_loops

def CS_with_weighted_sum_edge_weight_assign(preprocessed_combined_tensor, CS_threshold, coefficient):
    """
    Assigns edges and edge weights based on cosine similarity using combined image and non-image data.

    Parameters:
    - preprocessed_combined_tensor: A tensor containing combined image and non-image data.
    - CS_threshold: Threshold value to determine whether to create an edge between two nodes.
    - coefficient: Coefficient to combine image and non-image cosine similarity matrices.

    Returns:
    - edge_index: A tensor containing the edge indices of the constructed graph.
    - edge_weights: A tensor containing the edge weights corresponding to the edges.
    """
    num_nodes = preprocessed_combined_tensor.shape[0]
    img_tensor = preprocessed_combined_tensor[:, :-6]  # Image-related features
    nimg_tensor = preprocessed_combined_tensor[:, -6:]  # Non-image features

    edges = []
    edge_weights = []
    connected_nodes = set()

    # Calculate cosine similarity for image and non-image data separately
    CS_img = cosine_similarity(img_tensor)
    CS_nimg = cosine_similarity(nimg_tensor)

    # Combine the cosine similarity matrices using the given coefficient
    combined_CS = coefficient * CS_img + (1 - coefficient) * CS_nimg

    # Iterate over the upper triangular part of the matrix to assign edges
    for idx in range(num_nodes):
        for j in range(idx, num_nodes):
            if combined_CS[idx, j] >= CS_threshold:
                edges.append((idx, j))
                edge_weights.append(combined_CS[idx, j])
                # Add bidirectional edges except for self-loops
                if idx != j:
                    edges.append((j, idx))
                    edge_weights.append(combined_CS[idx, j])
                connected_nodes.update([idx, j])

    # Convert edges and weights to PyTorch tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    # Output information about the constructed graph
    print(f"CS_threshold = {CS_threshold}")
    print(f"Is the graph fully bidirectional? {'Yes' if is_undirected(edge_index) else 'No'}")
    print(f"Does the graph contain self-loops? {'Yes' if contains_self_loops(edge_index) else 'No'}")
    print(f"Number of isolated nodes: {num_nodes - len(connected_nodes)}")
    print(f"edge_index shape: {edge_index.shape}")
    print(f"edge_weights shape: {edge_weights.shape}")

    return edge_index, edge_weights


def euclidean_distance_matrix_scipy(tensor):
    """
    Calculate the Euclidean distance matrix using SciPy.

    Parameters:
    - tensor: A 2D tensor containing the feature vectors of nodes.

    Returns:
    - distances: A square matrix where each entry represents the Euclidean distance between two nodes.
    """
    dist_array = pdist(tensor, metric='euclidean')
    distances = squareform(dist_array)
    return distances

def ED_with_quantile_edge_weight_assign(preprocessed_combined_tensor, coefficient, quantile):
    """
    Assign edges based on Euclidean distance with all edge weights set to 1.

    Parameters:
    - preprocessed_combined_tensor: A tensor containing combined image and non-image data.
    - coefficient: Coefficient to combine image and non-image distance matrices.
    - quantile: Quantile value to determine the threshold for edge assignment.

    Returns:
    - edge_index: A tensor containing the edge indices of the constructed graph.
    - edge_weights: A tensor containing edge weights (all set to 1).
    """
    num_nodes = preprocessed_combined_tensor.shape[0]
    img_tensor = preprocessed_combined_tensor[:, :-6]  # Image-related features
    nimg_tensor = preprocessed_combined_tensor[:, -6:]  # Non-image features
    num_img_features = img_tensor.shape[1]
    num_nimg_features = nimg_tensor.shape[1]

    # Calculate Euclidean distance matrices for image and non-image data separately
    ED_img = euclidean_distance_matrix_scipy(img_tensor)
    ED_nimg = euclidean_distance_matrix_scipy(nimg_tensor)

    # Combine the distance matrices using the given coefficient and normalize by the number of features
    combined_ED = coefficient * (ED_img / num_img_features) + (1 - coefficient) * (ED_nimg / num_nimg_features)

    # Flatten the upper triangular part of the distance matrix and compute the threshold based on the quantile
    flat_distances = combined_ED[np.triu_indices(num_nodes, k=1)]
    threshold = np.percentile(flat_distances, quantile)
    print(f'Quantile: {quantile}')
    print(f'Threshold: {threshold}')

    edges = []
    edge_weights = []

    # Assign self-loops and edges based on the computed threshold
    for i in range(num_nodes):
        # Add self-loop
        edges.append((i, i))
        edge_weights.append(1)
        for j in range(i + 1, num_nodes):
            if combined_ED[i, j] <= threshold:
                # Add bidirectional edges
                edges.append((i, j))
                edges.append((j, i))
                edge_weights.append(1)  # All edge weights are set to 1
                edge_weights.append(1)  # Bidirectional edge weight

    # Convert edges and weights to PyTorch tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    # Output information about the constructed graph
    print(f"Is the graph fully bidirectional? {'Yes' if is_undirected(edge_index) else 'No'}")
    print(f"Does the graph contain self-loops? {'Yes' if contains_self_loops(edge_index) else 'No'}")
    print(f"Number of isolated nodes: {num_nodes - len(set(edge_index[0].tolist()))}")
    print(f"edge_index shape: {edge_index.shape}")
    print(f"edge_weights shape: {edge_weights.shape}")

    return edge_index, edge_weights