# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import time
import gc # For garbage collection
import multiprocessing # Import multiprocessing
import os # To check OS for DataLoader settings (optional)

# Try importing tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("tqdm library not found. Progress bars will not be shown.")
    # Define a dummy tqdm function if not available
    def tqdm(iterable, *args, **kwargs):
        desc = kwargs.get('desc', 'Processing items')
        print(f"{desc}...")
        result = list(iterable) # Consume the iterator to mimic processing
        print("... Done.")
        return result

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}") # Moved inside main block

# --- 1. Data Generation Function ---
def generate_standard_map_data(K, num_trajectories=2000, steps_per_traj=150, seed=None):
    """
    Generates data from the Standard Map dynamical system.

    Args:
        K (float): Chirikov parameter controlling the chaos level.
        num_trajectories (int): Number of distinct trajectories to generate.
        steps_per_traj (int): Number of time steps in each trajectory.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray: A numpy array of shape (num_trajectories, steps_per_traj, 2)
                    containing the (I, theta) coordinates for each step.
    """
    if seed is not None:
        np.random.seed(seed) # Set seed for reproducibility if provided

    data = []
    # Use tqdm for progress bar
    data_iterator = tqdm(range(num_trajectories), desc=f"Generating {num_trajectories} trajectories (seed={seed})", disable=not TQDM_AVAILABLE)
    for _ in data_iterator:
        # Initialize action (I) and angle (theta) randomly
        I = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, 2 * np.pi)
        trajectory = []
        for _ in range(steps_per_traj):
            # Standard Map equations
            I_new = (I + K * np.sin(theta)) % (2 * np.pi)
            theta_new = (theta + I_new) % (2 * np.pi)
            trajectory.append([I_new, theta_new])
            I, theta = I_new, theta_new
        data.append(trajectory)
    return np.array(data)

# --- 3. Data Preparation Function ---
def prepare_data(data, n_steps):
    """
    Prepares time series data for supervised learning by creating sequences.

    Args:
        data (np.ndarray): Input data of shape (num_trajectories, steps_per_traj, features).
        n_steps (int): The number of time steps to use as input features.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Input sequences (X) of shape (num_samples, n_steps, features).
            - np.ndarray: Target values (y) of shape (num_samples, features).
    """
    X, y = [], []
    # Use tqdm for progress bar
    prep_iterator = tqdm(data, desc="Preparing data sequences", disable=not TQDM_AVAILABLE)
    for traj in prep_iterator:
        # Slide a window of size n_steps across each trajectory
        for i in range(len(traj) - n_steps):
            X.append(traj[i:i + n_steps]) # Input sequence
            y.append(traj[i + n_steps])   # Target (next step)
    if not X: # Handle case where no sequences are generated
        print("Warning: No sequences generated during data preparation.")
        features = data.shape[2] if data.ndim == 3 and data.shape[2] > 0 else 1
        return np.empty((0, n_steps, features)), np.empty((0, features))
    return np.array(X), np.array(y)


# --- 4. Transformer Model Definition ---
class TransformerModel(nn.Module):
    """
    A Transformer Encoder model for time series forecasting.
    """
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=nn.GELU() # Using GELU activation
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() # Using GELU activation

    def forward(self, src):
        """ Forward pass of the model. """
        src = self.embedding(src) * np.sqrt(self.d_model)
        output = self.transformer_encoder(src)
        output = output[:, -1, :] # Use last time step
        output = self.dropout(self.activation(self.fc1(output)))
        output = self.fc2(output)
        return output

# --- 6. Training Loop ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    """ Trains the Transformer model with progress bars. """
    train_losses, val_losses = [], []
    print("Starting training...")
    start_time = time.time()

    # Outer loop for epochs with tqdm
    epoch_iterator = tqdm(range(epochs), desc="Epochs", disable=not TQDM_AVAILABLE)
    for epoch in epoch_iterator:
        epoch_start_time = time.time()
        model.train() # Set model to training mode
        running_train_loss = 0.0

        # Inner loop for training batches with tqdm
        train_batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False, disable=not TQDM_AVAILABLE)
        for i, (X_batch, y_batch) in enumerate(train_batch_iterator):
            # Data is already on device if using pin_memory and num_workers > 0
            # Otherwise, move manually: X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train_loss += loss.item()
            # Update tqdm postfix with current batch loss
            if TQDM_AVAILABLE:
                train_batch_iterator.set_postfix(loss=f"{loss.item():.4f}")

        # Ensure train_loader length is not zero before dividing
        if len(train_loader) > 0:
            avg_train_loss = running_train_loss / len(train_loader)
        else:
            avg_train_loss = 0.0
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        # Inner loop for validation batches with tqdm
        val_batch_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation", leave=False, disable=not TQDM_AVAILABLE)
        with torch.no_grad():
            for X_batch, y_batch in val_batch_iterator:
                # X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Move if needed
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item()
                if TQDM_AVAILABLE:
                     val_batch_iterator.set_postfix(loss=f"{loss.item():.4f}")

        # Ensure val_loader length is not zero before dividing
        if len(val_loader) > 0:
            avg_val_loss = running_val_loss / len(val_loader)
        else:
            avg_val_loss = 0.0
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss) # Step scheduler based on validation loss

        epoch_duration = time.time() - epoch_start_time
        log_message = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e} | Duration: {epoch_duration:.2f}s"
        print(log_message) # Print summary for the epoch
        # Update epoch iterator description
        if TQDM_AVAILABLE:
            epoch_iterator.set_description(f"Epochs (Val Loss: {avg_val_loss:.6f})")


    total_training_time = time.time() - start_time
    print(f"\nTraining finished in {total_training_time:.2f} seconds.")
    return train_losses, val_losses

# --- 8. Prediction Function ---
def predict_future_trajectory(model, initial_sequence, steps_to_predict, device, n_steps_input):
    """
    Predicts future steps of a trajectory using the trained model autoregressively.

    Args:
        model: The trained PyTorch model.
        initial_sequence (np.ndarray): The starting sequence of shape (n_steps, features).
        steps_to_predict (int): The number of future steps to predict.
        device: Device model is on.
        n_steps_input (int): Number of input steps model expects.

    Returns:
        np.ndarray: The predicted trajectory steps (excluding initial sequence),
                    shape (steps_to_predict, features).
    """
    model.eval()
    current_sequence = initial_sequence.copy()
    predicted_steps = []

    with torch.no_grad():
        # Add tqdm for prediction progress
        pred_iterator = tqdm(range(steps_to_predict), desc="Predicting future", leave=False, disable=not TQDM_AVAILABLE)
        for _ in pred_iterator:
            input_for_pred = current_sequence[-n_steps_input:]
            input_tensor = torch.FloatTensor(input_for_pred).unsqueeze(0).to(device)
            next_step_pred = model(input_tensor).cpu().numpy().squeeze(axis=0)
            predicted_steps.append(next_step_pred)
            # Append prediction reshaped correctly
            current_sequence = np.vstack([current_sequence, next_step_pred.reshape(1, -1)])

    return np.array(predicted_steps)


# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == '__main__':
    # Necessary for Windows multiprocessing support with spawn start method
    multiprocessing.freeze_support()

    print(f"Using device: {device}")

    # --- 2. Initial Data Generation and Visualization ---
    K = 0.2 # Chirikov parameter
    N_STEPS_INPUT = 20 # Number of past steps used to predict the next step

    # Generate initial data for training and initial testing
    print("Generating initial training/validation data...")
    initial_data = generate_standard_map_data(K, num_trajectories=2000, steps_per_traj=150, seed=42)
    print("Initial data shape:", initial_data.shape)

    # Visualize the initial training data sample characteristics
    print("Visualizing initial data...")
    plt.figure(figsize=(14, 10)) # Create figure for initial data plots

    # Plot a few random trajectories from the initial dataset
    plt.subplot(2, 2, 1)
    num_to_plot = min(5, len(initial_data))
    if len(initial_data) > 0:
        indices_to_plot = np.random.choice(len(initial_data), num_to_plot, replace=False)
        for i in indices_to_plot:
            plt.scatter(initial_data[i, :, 1], initial_data[i, :, 0], s=5, alpha=0.7, label=f'Traj {i}')
    plt.xlabel('θ (Angle)', fontsize=12)
    plt.ylabel('I (Action)', fontsize=12)
    plt.title(f'Random Trajectories (Initial Data, {num_to_plot} shown)', fontsize=14)
    plt.grid(alpha=0.3)
    if num_to_plot > 0: plt.legend(fontsize=8)


    # Plot density of all points in the initial dataset
    plt.subplot(2, 2, 2)
    all_points_initial = initial_data.reshape(-1, 2)
    plt.hist2d(all_points_initial[:, 1], all_points_initial[:, 0], bins=100, cmap='viridis')
    plt.xlabel('θ (Angle)', fontsize=12)
    plt.ylabel('I (Action)', fontsize=12)
    plt.title('Trajectory Density (Initial Data)', fontsize=14)
    plt.colorbar(label='Point Count')

    # Plot phase space distribution of the initial dataset
    plt.subplot(2, 2, 3)
    # Subsample for scatter plot if too many points
    points_to_scatter = min(all_points_initial.shape[0], 100000)
    if all_points_initial.shape[0] > 0:
        scatter_indices = np.random.choice(all_points_initial.shape[0], points_to_scatter, replace=False)
        plt.scatter(all_points_initial[scatter_indices, 1], all_points_initial[scatter_indices, 0], s=0.1, alpha=0.3)
    plt.xlabel('θ (Angle)', fontsize=12)
    plt.ylabel('I (Action)', fontsize=12)
    plt.title(f'Phase Space Distribution (Initial Data, {points_to_scatter} points)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.xlim(0, 2*np.pi)
    plt.ylim(0, 2*np.pi)

    # Plot time evolution of a single trajectory from the initial dataset
    plt.subplot(2, 2, 4)
    if len(initial_data) > 0:
        sample_traj_initial = initial_data[0]
        plt.plot(sample_traj_initial[:, 1], label='θ (Angle)')
        plt.plot(sample_traj_initial[:, 0], label='I (Action)')
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Time Evolution (Single Initial Trajectory)', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.suptitle(f'Initial Standard Map Data Visualization (K={K})', fontsize=16)
    # plt.show(block=False) # Changed to blocking show below

    # Prepare initial data for training and validation
    print(f"\nPreparing initial data with input sequence length {N_STEPS_INPUT}...")
    X, y = prepare_data(initial_data, N_STEPS_INPUT)
    print(f"Prepared initial data shape - X: {X.shape}, y: {y.shape}")
    del initial_data # Free memory
    gc.collect()

    # Split initial data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data shape - X: {X_train.shape}, y: {y_train.shape}")
    print(f"Validation data shape - X: {X_val.shape}, y: {y_val.shape}")
    del X, y # Free memory
    gc.collect()

    # Convert data to PyTorch tensors and move to the designated device (GPU or CPU)
    print("\nConverting data to Tensors...")
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    print("Tensor conversion complete.")
    gc.collect()

    # Create DataLoader instances for efficient batch processing
    BATCH_SIZE = 128
    # Set num_workers based on OS and available cores, but 0 often works best on Windows
    num_workers = 0 # Setting to 0 explicitly often avoids Windows issues
    pin_memory = True if device.type == 'cuda' and num_workers > 0 else False # pin_memory only works with num_workers > 0

    print(f"Using DataLoader with num_workers={num_workers}, pin_memory={pin_memory}")

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dataset = TensorDataset(X_val, y_val)
    # Use larger batch size for validation as gradients are not computed
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # --- 5. Model Initialization and Training Setup ---
    # Define model hyperparameters
    INPUT_DIM = 2 # (I, theta)
    D_MODEL = 128
    NHEAD = 8
    NUM_LAYERS = 4
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    EPOCHS = 50
    LEARNING_RATE = 1e-4

    # Instantiate the model and move it to the device
    model = TransformerModel(INPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT).to(device)
    print(f"\nModel Architecture:\n{model}\n")

    # Define the optimizer (AdamW) and loss function (MSE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False) # Set verbose=False to avoid warning

    # Show initial data plots (blocking)
    print("Displaying initial data plots. Close the plot window to continue...")
    plt.show() # Make the first plot blocking

    # Train the model
    if device.type == 'cuda': torch.cuda.empty_cache() # Clear cache before training
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, device)

    # Free up memory
    del X_train, y_train, X_val, y_val, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()

    # Plot training and validation loss curves
    plt.figure(figsize=(10, 5)) # Create new figure for loss plot
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Epochs', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.yscale('log')
    # plt.show(block=False) # Changed to blocking show below
    print("Displaying loss curve plot. Close the plot window to continue...")
    plt.show() # Make the loss plot blocking


    # --- 7. Regenerate Data for Final Evaluation ---
    print("\n--- Generating New Data for Final Evaluation ---")
    # Generate a completely new dataset using a different seed
    # Evaluation data size matches initial training data size (2000 trajectories, 150 steps)
    EVAL_TRAJECTORIES = 2000
    EVAL_STEPS = 150
    eval_data = generate_standard_map_data(K, num_trajectories=EVAL_TRAJECTORIES, steps_per_traj=EVAL_STEPS, seed=999) # Use a different seed
    print(f"Generated new evaluation data shape: {eval_data.shape}")

    # Prepare this new data using the same sequence length
    print(f"\nPreparing new evaluation data with input sequence length {N_STEPS_INPUT}...")
    X_eval, y_eval = prepare_data(eval_data, N_STEPS_INPUT)
    print(f"Prepared evaluation data shape - X_eval: {X_eval.shape}, y_eval: {y_eval.shape}")
    del eval_data # Free memory
    gc.collect()

    # Convert evaluation data to tensors
    print("\nConverting evaluation data to Tensors...")
    X_eval_tensor = torch.FloatTensor(X_eval).to(device)
    y_eval_tensor = torch.FloatTensor(y_eval).to(device)
    print("Tensor conversion complete.")
    gc.collect()

    # Create a DataLoader for the evaluation set (optional, can also sample directly)
    eval_dataset = TensorDataset(X_eval_tensor, y_eval_tensor)
    # Use larger batch size for evaluation
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


    # --- 9. Evaluate on a Sample Trajectory from New Data ---
    print("\n--- Evaluating Model on a Sample from New Evaluation Data ---")

    # Select a sample from the *new* evaluation data
    if len(X_eval) > 0:
        sample_index = np.random.randint(0, len(X_eval)) # Choose random index
        print(f"Using sample trajectory starting at index {sample_index} from new evaluation data.")
        initial_conditions_eval = X_eval[sample_index] # Shape: (N_STEPS_INPUT, 2)

        # Determine how many true future steps are available
        max_true_future_steps = len(y_eval) - sample_index
        STEPS_TO_PREDICT_EVAL = 100
        steps_available = min(STEPS_TO_PREDICT_EVAL, max_true_future_steps)

        if steps_available <= 0:
             print(f"Warning: Not enough subsequent data points available for sample index {sample_index}. Skipping single trajectory evaluation.")
        else:
            print(f"Predicting {steps_available} steps...")
            predicted_future_eval = predict_future_trajectory(model, initial_conditions_eval, steps_available, device, N_STEPS_INPUT)

            # Get the corresponding true future steps
            true_future_eval = y_eval[sample_index : sample_index + steps_available]

            # Combine for plotting
            full_true_traj_eval = np.vstack([initial_conditions_eval, true_future_eval])
            full_predicted_traj_eval = np.vstack([initial_conditions_eval, predicted_future_eval])

            print(f"Shape of true trajectory segment: {full_true_traj_eval.shape}")
            print(f"Shape of predicted trajectory segment: {full_predicted_traj_eval.shape}")


            # --- 10. Visualize Single Trajectory Prediction (New Data) ---
            print("Visualizing prediction comparison for the sample trajectory...")
            plt.figure(figsize=(15, 10)) # Create new figure for this plot

            # Plot phase space comparison
            plt.subplot(2, 2, 1)
            plt.scatter(full_true_traj_eval[:, 1], full_true_traj_eval[:, 0], s=15, label='True Trajectory (New Data)', c='blue', alpha=0.7)
            plt.scatter(full_predicted_traj_eval[:, 1], full_predicted_traj_eval[:, 0], s=15, label='Predicted Trajectory', c='red', alpha=0.7, marker='x')
            plt.scatter(initial_conditions_eval[:, 1], initial_conditions_eval[:, 0], s=20, label='Initial Conditions', c='green', alpha=0.9, marker='o')
            plt.xlabel('θ (Angle)', fontsize=12)
            plt.ylabel('I (Action)', fontsize=12)
            plt.title(f'Phase Space Comparison (New Eval Sample {sample_index})', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)
            plt.xlim(0, 2*np.pi); plt.ylim(0, 2*np.pi)

            # Plot I (Action) value over time
            plt.subplot(2, 2, 2)
            time_axis = np.arange(full_true_traj_eval.shape[0])
            plt.plot(time_axis, full_true_traj_eval[:, 0], 'b-', label='True I')
            plt.plot(time_axis, full_predicted_traj_eval[:, 0], 'r--', label='Predicted I')
            plt.axvline(x=N_STEPS_INPUT, color='gray', linestyle=':', label='Prediction Start')
            plt.xlabel('Time Step', fontsize=12); plt.ylabel('I (Action)', fontsize=12)
            plt.title('I Value Over Time (New Eval Data)', fontsize=14)
            plt.legend(fontsize=10); plt.grid(alpha=0.3)

            # Plot θ (Angle) value over time
            plt.subplot(2, 2, 3)
            plt.plot(time_axis, full_true_traj_eval[:, 1], 'b-', label='True θ')
            plt.plot(time_axis, full_predicted_traj_eval[:, 1], 'r--', label='Predicted θ')
            plt.axvline(x=N_STEPS_INPUT, color='gray', linestyle=':', label='Prediction Start')
            plt.xlabel('Time Step', fontsize=12); plt.ylabel('θ (Angle)', fontsize=12)
            plt.title('θ Value Over Time (New Eval Data)', fontsize=14)
            plt.legend(fontsize=10); plt.grid(alpha=0.3)

            # Plot prediction error over time
            plt.subplot(2, 2, 4)
            error_len = full_predicted_traj_eval.shape[0] - N_STEPS_INPUT
            if error_len > 0:
                error_I = np.abs(full_true_traj_eval[N_STEPS_INPUT:, 0] - full_predicted_traj_eval[N_STEPS_INPUT:, 0])
                error_theta = np.abs(full_true_traj_eval[N_STEPS_INPUT:, 1] - full_predicted_traj_eval[N_STEPS_INPUT:, 1])
                error_theta = np.minimum(error_theta, 2 * np.pi - error_theta) # Wrap angle error

                time_steps_pred = np.arange(error_len)
                plt.plot(time_steps_pred, error_I, 'b-', label='Absolute Error (I)')
                plt.plot(time_steps_pred, error_theta, 'r-', label='Absolute Error (θ)')
                plt.yscale('log')
                plt.xlabel('Prediction Step', fontsize=12); plt.ylabel('Absolute Error (log)', fontsize=12)
                plt.title('Prediction Error Over Time (New Eval Data)', fontsize=14)
                plt.legend(fontsize=10); plt.grid(alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No predicted steps', ha='center', va='center')
                plt.title('Prediction Error Over Time (New Eval Data)', fontsize=14)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.suptitle(f'Standard Map Trajectory Prediction on New Data (K={K}, Sample {sample_index})', fontsize=16)
            # plt.show(block=False) # Let the final plt.show() handle this
    else:
        print("Skipping single trajectory evaluation as X_eval is empty.")


    # --- 11. Quantitative Evaluation on New Data ---
    print("\n--- Quantitative Evaluation on New Evaluation Data ---")

    # Evaluate overall MSE on the *entire* new evaluation set (one-step prediction)
    model.eval()
    all_outputs_eval = []
    all_targets_eval = []
    print("Calculating one-step prediction MSE on the entire new evaluation set...")
    eval_start_time = time.time()
    # Use tqdm for evaluation loader
    eval_batch_iterator = tqdm(eval_loader, desc="Evaluating one-step MSE", leave=False, disable=not TQDM_AVAILABLE)
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(eval_batch_iterator):
            # Data already on device if using pin_memory and num_workers > 0
            outputs = model(X_batch)
            all_outputs_eval.append(outputs.cpu().numpy())
            all_targets_eval.append(y_batch.cpu().numpy())

    if all_outputs_eval and all_targets_eval:
        all_outputs_eval = np.concatenate(all_outputs_eval, axis=0)
        all_targets_eval = np.concatenate(all_targets_eval, axis=0)
        eval_duration = time.time() - eval_start_time
        print(f"One-step evaluation finished in {eval_duration:.2f} seconds.")
        one_step_mse_eval = mean_squared_error(all_targets_eval, all_outputs_eval)
        print(f"One-Step Prediction MSE on entire new evaluation set ({len(all_targets_eval)} samples): {one_step_mse_eval:.6f}")
    else:
        print("Warning: No samples processed during one-step evaluation.")


    # Evaluate multi-step prediction performance
    NUM_TRAJECTORIES_TO_TEST = 50
    PREDICTION_HORIZON = 50

    all_rmse_I = []
    all_rmse_theta = []
    all_corr_I = []
    all_corr_theta = []

    print(f"\nCalculating multi-step ({PREDICTION_HORIZON} steps) prediction metrics over {NUM_TRAJECTORIES_TO_TEST} new trajectories...")
    multi_step_eval_start_time = time.time()

    num_to_actually_test = min(NUM_TRAJECTORIES_TO_TEST, len(X_eval))
    if num_to_actually_test == 0:
        print("Warning: No evaluation samples available (X_eval is empty). Skipping multi-step evaluation.")
    else:
        eval_indices = np.random.choice(len(X_eval), num_to_actually_test, replace=False)
        processed_count = 0
        # Use tqdm for trajectory testing loop
        traj_test_iterator = tqdm(eval_indices, desc="Testing multi-step prediction", disable=not TQDM_AVAILABLE)
        for i in traj_test_iterator:
            initial_sequence = X_eval[i]
            if i + PREDICTION_HORIZON > len(y_eval):
                continue # Skip if not enough future steps

            true_future = y_eval[i : i + PREDICTION_HORIZON]
            predicted_future = predict_future_trajectory(model, initial_sequence, PREDICTION_HORIZON, device, N_STEPS_INPUT)

            if predicted_future.shape[0] != PREDICTION_HORIZON:
                 continue # Skip if prediction length mismatch

            # Calculate metrics
            rmse_I = np.sqrt(mean_squared_error(true_future[:, 0], predicted_future[:, 0]))
            diff_theta = true_future[:, 1] - predicted_future[:, 1]
            diff_theta = (diff_theta + np.pi) % (2 * np.pi) - np.pi # Wrap difference
            rmse_theta = np.sqrt(np.mean(diff_theta**2))

            # Calculate correlations safely
            if np.std(true_future[:, 0]) > 1e-6 and np.std(predicted_future[:, 0]) > 1e-6:
                 try:
                     corr_I, _ = pearsonr(true_future[:, 0], predicted_future[:, 0])
                     if not np.isnan(corr_I): all_corr_I.append(corr_I)
                 except ValueError: pass
            if np.std(true_future[:, 1]) > 1e-6 and np.std(predicted_future[:, 1]) > 1e-6:
                 try:
                     corr_theta, _ = pearsonr(true_future[:, 1], predicted_future[:, 1])
                     if not np.isnan(corr_theta): all_corr_theta.append(corr_theta)
                 except ValueError: pass

            all_rmse_I.append(rmse_I)
            all_rmse_theta.append(rmse_theta)
            processed_count += 1

        multi_step_eval_duration = time.time() - multi_step_eval_start_time
        print(f"Multi-step evaluation finished in {multi_step_eval_duration:.2f} seconds.")

        # Calculate and print average metrics
        avg_rmse_I = np.mean(all_rmse_I) if all_rmse_I else np.nan
        std_rmse_I = np.std(all_rmse_I) if all_rmse_I else np.nan
        avg_rmse_theta = np.mean(all_rmse_theta) if all_rmse_theta else np.nan
        std_rmse_theta = np.std(all_rmse_theta) if all_rmse_theta else np.nan
        avg_corr_I = np.mean(all_corr_I) if all_corr_I else np.nan
        std_corr_I = np.std(all_corr_I) if all_corr_I else np.nan
        avg_corr_theta = np.mean(all_corr_theta) if all_corr_theta else np.nan
        std_corr_theta = np.std(all_corr_theta) if all_corr_theta else np.nan

        print(f"\nAverage Multi-Step ({PREDICTION_HORIZON} steps) Prediction Metrics ({processed_count} valid trajectories):")
        print(f"  RMSE (I):     {avg_rmse_I:.6f} ± {std_rmse_I:.6f}")
        print(f"  RMSE (θ):     {avg_rmse_theta:.6f} ± {std_rmse_theta:.6f} (Angle diff wrapped)")
        print(f"  Corr (I):     {avg_corr_I:.4f} ± {std_corr_I:.4f}")
        print(f"  Corr (θ):     {avg_corr_theta:.4f} ± {std_corr_theta:.4f} (Simple Pearson)")

    # Final cleanup
    del X_eval, y_eval, X_eval_tensor, y_eval_tensor, eval_loader, eval_dataset
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()

    print("\nEvaluation complete.")

    # Keep ALL plots open until user closes them
    print("Displaying final plots. Close plot windows to exit.")
    plt.show() # Final blocking call to show all generated figures

