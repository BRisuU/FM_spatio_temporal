import numpy as np
import torch

def compute_batch_similarity(Y_batch, threshold=0.99):
    if Y_batch.size(-1) == 1:
        Y_batch = Y_batch.squeeze(-1)
    Y_batch = Y_batch.view(-1, Y_batch.size(1))  # (batch_size * num_sensors, output_dim)

    batch_pairs = []
    sim_matrix = cosine_similarity_torch(Y_batch)
    for i in range(len(Y_batch)):
        for j in range(i + 1, len(Y_batch)):
            sim = sim_matrix[i, j].item()
            if sim >= threshold:
                batch_pairs.append((i, j))
    return batch_pairs
                    
def cosine_similarity_numpy(x):
    # x: shape (N, D)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_normalized = x / (x_norm + 1e-10) # Prevent division by zero
    sim_matrix = np.dot(x_normalized, x_normalized.T)
    return sim_matrix

def cosine_similarity_torch(x):
        # x: (N, D)
        x_norm = torch.norm(x, dim=1, keepdim=True)  # (N, 1)
        x_normalized = x / (x_norm + 1e-10)          # (N, D)
        sim_matrix = x_normalized @ x_normalized.t() # (N, N)
        return sim_matrix