import time
import json
import os
import numpy as np
import torch
import torch.nn as nn
import tempfile
import shutil
from torch.nn import functional as F

class BaseModel(nn.Module):
    def __init__(self, model_name, config=None):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.config = config if config is not None else {}
        self.trained = False
        self.loss_history = []  # Track loss development during training
        self.train_time = None
        self.inference_time = None
        self.communication_time = None
        self.y_pred = None

    def train_model(self, X_train, Y_train, Y_train_mask=None):
        """Train the model (to be implemented by child classes)."""
        raise NotImplementedError("train method must be implemented in the child class.")

    def predict(self, X_test):
        """Make predictions (to be implemented by child classes)."""
        raise NotImplementedError("predict method must be implemented in the child class.")

    def evaluate(self, Y_true, Y_pred):
        """Evaluate the model's performance."""
        mse = ((Y_true - Y_pred) ** 2).mean()
        mae = abs(Y_true - Y_pred).mean()
        return {"MSE": mse, "MAE": mae}
    
    def masked_mse(self, y_true, y_pred, mask):
        error = (y_true - y_pred)**2
        masked_error = error * mask  # Zero out missing values
        return masked_error.sum() / mask.sum()  # Normalize by valid entries
    
    def masked_mae(self, y_true, y_pred, mask):
        error = abs(y_true - y_pred)
        masked_error = error * mask
        return masked_error.sum() / mask.sum()  # Normalize by valid entries
    
    def masked_r2(self, y_true, y_pred, mask):
        # Calculate R^2 score with masking
        ss_res = ((y_true - y_pred) ** 2 * mask).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2 * mask).sum()
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    
    def masked_rmse(self, y_true, y_pred, mask):
        # Calculate RMSE with masking
        mse = self.masked_mse(y_true, y_pred, mask)
        return torch.sqrt(mse)
    
    def masked_mape(self, y_true, y_pred, mask, epsilon=1e-8):
        error = torch.abs((y_true - y_pred) / (y_true + epsilon))
        masked_error = error * mask
        return masked_error.sum() / mask.sum() * 100

    
    def nt_xent_loss(self, temporal_emb, similarity_pairs, temperature=0.5):
        """
        temporal_emb: Tensor of shape (batch_size, num_sensors, seq_len, hidden_dim)
        similarity_pairs: List of tuples [((i1, j1), (i2, j2)), ...] where i = sample index, j = sensor index
        """
        loss = 0.0
        num_pairs = 0

        for (i, j) in similarity_pairs:
            emb1 = temporal_emb[i]  # shape: (seq_len, hidden_dim)
            emb2 = temporal_emb[j]  # shape: (seq_len, hidden_dim)

            # Normalize
            emb1_flat = F.normalize(emb1, dim=0)
            emb2_flat = F.normalize(emb2, dim=0)

            # Compute cosine similarity
            sim_pos = torch.dot(emb1_flat, emb2_flat) / temperature

            # Compute negatives (optional: sample a few negatives for efficiency)
            #all_embeddings = temporal_emb.view(-1, temporal_emb.size(2), temporal_emb.size(3))  # (batch_size * num_sensors, seq_len, hidden_dim)
            all_flat = F.normalize(temporal_emb)  # (N, seq_len * hidden_dim)

            sim_all = torch.matmul(emb1_flat.unsqueeze(0), all_flat.T).squeeze(0) / temperature
            sim_all = torch.exp(sim_all)

            numerator = torch.exp(sim_pos)
            denominator = sim_all.sum() - numerator  # exclude positive pair

            loss += -torch.log(numerator / denominator)
            num_pairs += 1

        return loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=temporal_emb.device)

    
    def remove_single_dimensions(self, np_array):
        """Remove single-dimensional entries from the shape of an array."""
        # Check if the input array has 4 dimensions
        if np_array.ndim == 4:
            # Check if the last dimension is 1 and the others are greater than 1
            if np_array.shape[-1] == 1:
                # Remove the last dimension
                np_array_reduced = np_array.squeeze(-1)
                #print(f"Original shape: {np_array.shape}, Reduced shape: {np_array_reduced.shape}")
                return np_array_reduced
            else:
                print("Dimension reduction not possiblw")
        else:
            return np_array
    
    def store_predictions(self, Y_pred, Y_true, Y_mask, file_path):
        """Store predictions to a file."""
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        #torch.save(Y_pred, file_path.replace(".npz", ".pth"))
        # Safe as csv
        Y_pred = Y_pred.cpu().numpy()
        Y_true = Y_true.cpu().numpy()
        Y_mask = Y_mask.cpu().numpy()
        
        Y_pred = self.remove_single_dimensions(Y_pred)
        Y_true = self.remove_single_dimensions(Y_true)
        Y_mask = self.remove_single_dimensions(Y_mask)
        
        # Shape samples, predicted time steps, sensors
        # Store it as np array pred, samples, time steps, sensors
        np.savez(file_path, Y_pred=Y_pred, Y_true=Y_true, Y_mask=Y_mask)

        
    
    def denormalize(self, norm_params):
        if norm_params:
            norm_params = {k: v.to(self.device) for k, v in norm_params.items()}
           
            # min-max normalization
            diff = norm_params['X_max'] - norm_params['X_min']
            Y_min = norm_params['X_min']
            # Reshape using None (numpy-style newaxis)
            #self.Y_pred = self.Y_pred * diff[None, None, :, None] + Y_min[None, None, :, None]
            
            ndim = self.Y_pred.dim()
            # Create broadcast shape: [1]*(k) + [C] + [1]*(ndim-k-1)
            shape = [1] * ndim
            shape[2] = -1
            # Reshape normalization parameters
            diff_reshaped = diff.view(*shape)
            Y_min_reshaped = Y_min.view(*shape)

            # Apply normalization
            self.Y_pred = self.Y_pred * diff_reshaped + Y_min_reshaped
        

    def evaluate_and_store(self, train_loader, test_loader, test_data, y_true=None, y_true_mask=None, normalize=False, norm_params=None, results_file="results_server/results.json", fl=False, adjacency_matrix=None, test_loader_exogenous=None):
        """Train, evaluate, and store results to a centralized file."""
        y_true = test_data.Y_raw
        y_true_mask = test_data.Y_mask
        norm_params = test_data.norm_params
        
        if not fl:
            # Train the model
            start_time = time.time()
            self.train_model(train_loader)
            self.train_time = time.time() - start_time
                
            # Store model parameters
            path_params = f"results/trained_models/{self.model_name}.pth"
            #self.store_model(path_params)

        # Predict
        start_time = time.time()
        if adjacency_matrix is not None:
            adjacency_matrix = adjacency_matrix.to(self.device)
            self.Y_pred = self.predict(test_loader, adjacency_matrix, test_loader_exogenous)
        else:
            self.Y_pred = self.predict(test_loader)

        self.inference_time = time.time() - start_time
        #if normalize:
        self.denormalize(norm_params)
        
            
        # Evaluate metrics
        """
        #metrics = self.evaluate(Y_test, Y_pred)
        Y_test = [Y for _, Y, _, _ in test_loader]  # Extract Y values from test_loader
        Y_test = torch.cat(Y_test, dim=0)  # Combine all batches
        Y_test_mask = [Y_mask for _, _, _, Y_mask in test_loader]
        Y_test_mask = torch.cat(Y_test_mask, dim=0)  # Combine all masks
        """
        # Check the length of y_true and y_true_mask compared to Y_pred
        if y_true.shape[0] != self.Y_pred.shape[0]:
            # Remove the last elements to match the lengths
            min_length = min(y_true.shape[0], self.Y_pred.shape[0])
            y_true = y_true[:min_length]
            y_true_mask = y_true_mask[:min_length]
            print(f"Warning: Adjusted y_true and y_true_mask to match Y_pred length: {min_length}")
        
        # all data to cpu
        y_true = y_true.cpu()
        y_true_mask = y_true_mask.cpu()
        self.Y_pred = self.Y_pred.cpu()
        
        #print(f"Type of y_true: {type(y_true)}, Type of y_true_mask: {type(y_true_mask)}, Type of Y_pred: {type(self.Y_pred)}")
        
        #print(f"Shape of y_true: {y_true.shape}, Shape of y_true_mask: {y_true_mask.shape}, Shape of Y_pred: {self.Y_pred.shape}")
        
        
        metrics = {
            "MSE": self.masked_mse(y_true, self.Y_pred, y_true_mask).item(),
            "MAE": self.masked_mae(y_true, self.Y_pred, y_true_mask).item(),
            "RMSE": self.masked_rmse(y_true, self.Y_pred, y_true_mask).item(),
            "R2": self.masked_r2(y_true, self.Y_pred, y_true_mask).item(),
            "MAPE": self.masked_mape(y_true, self.Y_pred, y_true_mask).item()
        }
        # "MAE": self.masked_mae(Y_test, Y_pred, Y_test_mask).item(),
        
        self.store_predictions(self.Y_pred, y_true, y_true_mask, f"results_server/predictions/{self.model_name}_predictions.npz")
        
        np_file = np.load(f"results_server/predictions/{self.model_name}_predictions.npz")
        #print(f"shape of predictions: {np_file['Y_pred'].shape}")
        

        # Prepare results
        results = {
            "model_name": self.model_name,
            "creation_time": "2026-01-22 10:16:50",
            "train_time": self.train_time,
            "inference_time": self.inference_time,
            "metrics": metrics,
            "config": self.config,
            "loss_history": self.loss_history,  # Include loss development
            "number_of_parameters": sum(p.numel() for p in self.parameters())
        }
        
        # Drop adj is key in self.config dict if it exist
        if 'adj' in self.config:
            del self.config['adj']
            
        
        def safe_json_write(results, results_file):
            # Save results to centralized file
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    try:
                        all_results = json.load(f)
                    except json.JSONDecodeError:
                        print("Warning: Corrupted JSON. Starting fresh.")
                        #all_results = {}
            else:
                all_results = {}
                # Create the path and file if it does not exist
                os.makedirs(os.path.dirname(results_file), exist_ok=True)

            # Convert torch tensors
            for key, value in results.items():
                if isinstance(value, torch.Tensor):
                    results[key] = value.tolist()

            # Update results
            all_results[self.model_name] = results

            # Safe write using temp file
            try:
                with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
                    json.dump(all_results, tmp, indent=2)
                    temp_name = tmp.name
                shutil.move(temp_name, results_file)
            except Exception as e:
                print(f"Failed to save results: {e}")

        safe_json_write(results, results_file)

        #print(f"Results for {self.model_name} saved to {results_file}")
        



    def save_model(self, folder_path="results_server/trained_models"):
        
        """Save the trained model to a file."""
        if not self.trained:
            raise ValueError("Model must be trained before saving.")
        with open(f"{folder_path}/{self.model_name}.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(file_path="results/trained_models"):
        """Load a saved model from a file."""
        with open(f"{folder_path}/{self.model_name}", "rb") as f:
            return pickle.load(f)