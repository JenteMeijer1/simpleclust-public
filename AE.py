"""Train and evaluate deterministic autoencoders for modality reduction."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import dill


import os


from skopt import gp_minimize
from skopt.space import Categorical


# ─── Model definition ─────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim,
                 dropout_rate=0.1, activation_fn=nn.ReLU(), l2_reg=1e-3):
        """Initialize the object."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
            activation_fn,
            nn.Dropout(dropout_rate),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
        )
        self.l2_reg = l2_reg

    def forward(self, x):
        """Handle forward."""
        code = self.encoder(x)
        return self.decoder(code)

    def l2_loss(self):
        """Handle l2 loss."""
        return self.l2_reg * sum(torch.norm(p) for p in self.parameters())

    def encode(self, x):
        """Handle encode."""
        return self.encoder(x)


# ─── Training & evaluation helpers ──────────────────────────────────────────
def _latent_l1_loss(model, xb, l1_reg):
    """Handle latent l1 loss."""
    if not l1_reg:
        return xb.new_tensor(0.0)
    return float(l1_reg) * model.encode(xb).abs().mean()


def train_model(model, dataloader, num_epochs, lr, device, l1_reg=0.0):
    """Train model."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(num_epochs):
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            output = model(xb)
            loss = criterion(output, yb) + model.l2_loss() + _latent_l1_loss(model, xb, l1_reg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate_model(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds.append(out.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    return mean_squared_error(trues, preds), preds

# Early stopping training helper
def train_model_es(model, train_loader, val_loader, num_epochs, lr, device, patience=15, l1_reg=0.0):
    """Train model es."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    epochs_no_improve = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb) + model.l2_loss() + _latent_l1_loss(model, xb, l1_reg)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_losses.append((criterion(out, yb) + model.l2_loss() + _latent_l1_loss(model, xb, l1_reg)).item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)


# ─── Nested CV with Bayesian search ─────────────────────────────────────────
def run_nested_cv_autoencoder(
    X,
    hidden_dims,
    activation_functions,
    learning_rates,
    batch_sizes,
    latent_dims,
    outer_splits=5,
    inner_splits=3,
    device='cpu',
    l1_reg=0.0
):
    """Run nested cv autoencoder."""
    import os as _os
    from joblib import Parallel as _Parallel, delayed as _delayed
    # Use all logical CPUs on this node
    total_cores = _os.cpu_count() or 1
    outer_jobs = min(outer_splits, total_cores)
    inner_jobs = max(1, total_cores // outer_jobs)

    # Change latent dims if data dimensionality is smaller than the latent dim
    if X.shape[1] < max(latent_dims):
        latent_dims = [2, 5, 10]
        if X.shape[1] < max(latent_dims):
            latent_dims = [2, 5]
            if X.shape[1] < max(latent_dims):
                latent_dims = [2]

    outer_kf = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    results = []
    all_true, all_pred = [], []
    all_train_true, all_train_pred = [], []
    all_test_latents, all_train_latents = [], []

    # Helper function for processing a single CV fold in parallel
    def _process_fold(fold, train_idx, test_idx):
        """Process fold."""
        X_train, X_test = X[train_idx], X[test_idx]
        input_dim = X.shape[1]

        # Build search space over names, so we can record the activation key
        space = [
            Categorical(hidden_dims, name='hidden_dim'),
            Categorical(latent_dims, name='latent_dim'),
            Categorical(list(activation_functions.keys()), name='activation_name'),
            Categorical(learning_rates, name='learning_rate'),
            Categorical(batch_sizes, name='batch_size'),
        ]

        def objective(vals):
            """Handle objective."""
            hdim, ldim, act_name, lr, bs = vals
            act_fn = activation_functions[act_name]
            inner_kf = KFold(n_splits=inner_splits, shuffle=True, random_state=42)
            def _inner_eval(i_tr, i_val):
                """Handle inner eval."""
                X_tr, X_val = X_train[i_tr], X_train[i_val]
                mdl = Autoencoder(
                    input_dim=input_dim,
                    hidden_dim=hdim,
                    latent_dim=ldim,
                    dropout_rate=0.1,
                    activation_fn=act_fn,
                    l2_reg=1e-3
                )
                tr_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(X_tr))
                tr_ld = DataLoader(tr_ds, batch_size=int(bs), shuffle=True)
                val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(X_val))
                val_ld = DataLoader(val_ds, batch_size=int(bs), shuffle=False)
                train_model_es(mdl, tr_ld, val_ld, num_epochs=150, lr=lr, device=device, patience=15, l1_reg=l1_reg)
                loss, _ = evaluate_model(mdl, val_ld, device)
                return loss
            # Parallel inner‐CV across splits
            losses = _Parallel(n_jobs=inner_jobs)(
                _delayed(_inner_eval)(i_tr, i_val)
                for i_tr, i_val in inner_kf.split(X_train)
            )
            return float(np.mean(losses))

        from skopt import gp_minimize
        res = gp_minimize(objective, space, n_calls=20, random_state=42)

        best_h, best_l, best_act_name, best_lr, best_bs = res.x

        # Train final model on full X_train
        final = Autoencoder(
            input_dim=input_dim,
            hidden_dim=best_h,
            latent_dim=best_l,
            dropout_rate=0.1,
            activation_fn=activation_functions[best_act_name],
            l2_reg=1e-3
        )
        full_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(X_train))
        full_ld = DataLoader(full_ds, batch_size=int(best_bs), shuffle=True)
        train_model(final, full_ld, num_epochs=150, lr=best_lr, device=device, l1_reg=l1_reg)

        # Ensure directory for saved models exists
        #os.makedirs(f"saved_models", exist_ok=True)
        #torch.save({
        #    "hidden_dim": best_h,
        #    "latent_dim": best_l,
        #    "activation_name": best_act_name,
        #    "state_dict": final.state_dict()
        #}, f"saved_models/ae_final_fold{fold}.pt")

        # Evaluate on train and test
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(X_train))
        test_ds  = TensorDataset(torch.FloatTensor(X_test),  torch.FloatTensor(X_test))
        train_ld = DataLoader(train_ds, batch_size=int(best_bs), shuffle=False)
        test_ld  = DataLoader(test_ds,  batch_size=int(best_bs), shuffle=False)
        train_loss, train_pred = evaluate_model(final, train_ld, device)
        test_loss,  test_pred  = evaluate_model(final, test_ld,  device)

        with torch.no_grad():
            train_latent = final.encode(torch.FloatTensor(X_train).to(device)).cpu().numpy()
            test_latent  = final.encode(torch.FloatTensor(X_test).to(device)).cpu().numpy()

        fold_result = {
            'fold': fold,
            'test_loss': test_loss,
            'params': {
                'hidden_dim': best_h,
                'latent_dim': best_l,
                'activation': best_act_name,
                'learning_rate': best_lr,
                'batch_size': best_bs
            }
        }

        return fold_result, X_test, test_pred, X_train, train_pred, test_latent, train_latent

    # Parallel execution over folds
    fold_splits = list(enumerate(outer_kf.split(X), start=1))
    parallel_outputs = _Parallel(n_jobs=outer_jobs)(
        _delayed(_process_fold)(fold, train_idx, test_idx)
        for fold, (train_idx, test_idx) in fold_splits
    )
    for fold_result, true_arr, pred_arr, train_arr, train_pred_arr, test_latent, train_latent in parallel_outputs:
        results.append(fold_result)
        all_true.append(true_arr)
        all_pred.append(pred_arr)
        all_train_true.append(train_arr)
        all_train_pred.append(train_pred_arr)
        all_test_latents.append(test_latent)
        all_train_latents.append(train_latent)

    return (
        results,
        np.vstack(all_true),
        np.vstack(all_pred),
        np.vstack(all_test_latents),
        np.vstack(all_train_latents),
    )


# ─── Wrapper to accept a DataFrame or array ───────────────────────────────────
def prepare_and_run_autoencoder(df, device='cpu', hidden_dims=[128, 256, 512], activation_functions = {'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(), 'selu': nn.SELU(), 'swish': nn.SiLU()}, learning_rates=[0.001, 0.0001], batch_sizes=[32, 64, 128], latent_dims=[2, 5, 10], l1_reg=0.0):
    """Prepare and run autoencoder."""
    X = df.values if isinstance(df, pd.DataFrame) else df
    return run_nested_cv_autoencoder(
        X,
        hidden_dims=hidden_dims,
        activation_functions=activation_functions,
        learning_rates=learning_rates,
        batch_sizes=batch_sizes,
        latent_dims = latent_dims,
        device=device,
        l1_reg=l1_reg
    )


def run_all_modalities(modalities_dict, device='cpu', split = 'discovery',
                                        activation_functions = {'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(), 'selu': nn.SELU(), 'swish': nn.SiLU()}):
    """
    For each modality in modalities_dict (name → DataFrame or array),
    runs prepare_and_run_autoencoder and collects the train+test latent codes.

    Returns:
      latent_vars: dict mapping modality name to {
        'train_latent': np.ndarray,
        'test_latent':  np.ndarray
      }
      results_by_modality: dict mapping modality name to the full `results` list
                         (fold‐wise summaries)
    """
    latent_vars = {}



    for modality, df in modalities_dict.items():
        print(f"▶️  Running autoencoder for modality: {modality}")

        ids, matrix = df



        # Unpack the 7 outputs; we only need the 1st and last two here
        results, all_true, all_pred, test_latent, train_latent = \
            prepare_and_run_autoencoder(matrix, device=device,
                                        hidden_dims=[128, 256, 512],
                                        activation_functions = {'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(), 'selu': nn.SELU(), 'swish': nn.SiLU()},
                                        learning_rates=[0.001, 0.0001],
                                        batch_sizes=[32, 64, 128],
                                        latent_dims=[2, 5, 10])


        # Select best hyperparameters (minimum test_loss)
        best = min(results, key=lambda r: r['test_loss'])['params']

        X = matrix.values if isinstance(df, pd.DataFrame) else matrix

        # Initialize final model
        final = Autoencoder(
            input_dim=X.shape[1],
            hidden_dim=best['hidden_dim'],
            latent_dim=best['latent_dim'],
            activation_fn=activation_functions[best['activation']],
            dropout_rate=0.1,
            l2_reg=1e-3
        )

        # Train on full data set
        full_ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(X))
        full_ld = DataLoader(full_ds, batch_size=int(best['batch_size']), shuffle=True)
        train_model(final, full_ld, num_epochs=150, lr=best['learning_rate'], device=device)

        # Save the final model
        #os.makedirs(f"saved_models_{modality}", exist_ok=True)
        #torch.save({
        #    "hidden_dim": best['hidden_dim'],
        #    "latent_dim": best['latent_dim'],
        #    "activation_name": best['activation'],
        #    "state_dict": final.state_dict()
        #}, f"saved_models_{modality}/ae_final_full_discovery.pt")
        # Encode discovery data
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            discovery_latents = final.encode(X_tensor).cpu().numpy()


        # Store just the latents in a dict
        latent_vars = {
            'train_latent': train_latent,
            'test_latent':  test_latent
        }


        # Save results
        results_one_modality = {
            'ids': ids,
            'results': results,
            'all_true': all_true,
            'all_pred': all_pred,
            'latent_vars': latent_vars,
            'final_latent': discovery_latents
        }
        with open(f"ae_results_{modality}_{split}.pkl", "wb") as f:
            dill.dump(results_one_modality, f)
        print("Results saved.")




def load_and_encode(model_path, X_new, device="cpu", activation_functions = {'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(), 'selu': nn.SELU(), 'swish': nn.SiLU()}):
    # Load full checkpoint (not just weights)
    """Load and encode."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Reconstruct the same architecture
    model = Autoencoder(
        input_dim = X_new.shape[1],
        hidden_dim = ckpt["hidden_dim"],
        latent_dim = ckpt["latent_dim"],
        activation_fn = activation_functions[ckpt["activation_name"]],
        dropout_rate = 0.1,
        l2_reg = 1e-3
    )

    # Load weights
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device).eval()

    # Encode new data
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_new).to(device)
        return model.encode(X_tensor).cpu().numpy()




def run_all_modalities_test(modalities_dict, device='cpu', split = 'test',
                                        hidden_dims=[128, 256, 512],
                                        activation_functions = {'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(), 'selu': nn.SELU(), 'swish': nn.SiLU()},
                                        learning_rates=[0.001, 0.0001],
                                        batch_sizes=[32, 64, 128],
                                        latent_dims=[2, 5, 10]):
    """
    For each modality in modalities_dict (name → DataFrame or array),
    runs prepare_and_run_autoencoder and collects the train+test latent codes.

    Returns:
      latent_vars: dict mapping modality name to {
        'train_latent': np.ndarray,
        'test_latent':  np.ndarray
      }
      results_by_modality: dict mapping modality name to the full `results` list
                         (fold‐wise summaries)
    """

    for modality, df in modalities_dict.items():
        print(f"▶️  Running autoencoder for modality: {modality}")

        ids, matrix = df

        X = matrix.values if isinstance(df, pd.DataFrame) else matrix

        latents_test = load_and_encode(f"saved_models_{modality}/ae_final_full_discovery.pt", X, activation_functions = activation_functions)



        # Save results
        results_one_modality = {
            'ids': ids,
            'latent_vars': latents_test
        }
        with open(f"ae_results_{modality}_{split}.pkl", "wb") as f:
            dill.dump(results_one_modality, f)
        print("Results saved.")







def run_AE_complete (modalities_dict, device='cpu', split = 'discovery', hidden_dims = [128, 256, 512], activation_functions = {'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(), 'selu': nn.SELU(), 'swish': nn.SiLU()}, learning_rates = [0.001, 0.0001], batch_sizes = [32, 64, 128], latent_dims = [2,5,10], l1_reg=0.0):
    """
    For each modality in modalities_dict (name → DataFrame or array),
    runs prepare_and_run_autoencoder and collects the train+test latent codes.

    Returns:
      latent_vars: dict mapping modality name to {
        'train_latent': np.ndarray,
        'test_latent':  np.ndarray
      }
      results_by_modality: dict mapping modality name to the full `results` list
                         (fold‐wise summaries)
    """

    results_by_modality = {}
    latent_vars = {}
    for modality, df in modalities_dict.items():
        print(f"▶️  Running autoencoder for modality: {modality}")
        ids, matrix = df

        # Unpack the 7 outputs; we only need the 1st and last two here
        results, all_true, all_pred, test_latent, train_latent = \
            prepare_and_run_autoencoder(matrix, device=device,
                                        hidden_dims=hidden_dims,
                                        activation_functions = activation_functions,
                                        learning_rates=learning_rates,
                                        batch_sizes=batch_sizes,
                                        latent_dims=latent_dims,
                                        l1_reg=l1_reg)

        # Select best hyperparameters (minimum test_loss)
        best = min(results, key=lambda r: r['test_loss'])['params']

        X = matrix.values if isinstance(df, pd.DataFrame) else matrix

        # Initialize final model
        final = Autoencoder(
            input_dim=X.shape[1],
            hidden_dim=best['hidden_dim'],
            latent_dim=best['latent_dim'],
            activation_fn=activation_functions[best['activation']],
            dropout_rate=0.1,
            l2_reg=1e-3
        )

        # Train on full data set
        full_ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(X))
        full_ld = DataLoader(full_ds, batch_size=int(best['batch_size']), shuffle=True)
        train_model(final, full_ld, num_epochs=150, lr=best['learning_rate'], device=device, l1_reg=l1_reg)

        # Save the final model
        #os.makedirs(f"saved_models_{modality}", exist_ok=True)
        #torch.save({
        #    "hidden_dim": best['hidden_dim'],
        #    "latent_dim": best['latent_dim'],
        #    "activation_name": best['activation'],
        #    "state_dict": final.state_dict()
        #}, f"saved_models_{modality}/ae_final_full_discovery.pt")
        # Encode discovery data
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            discovery_latents = final.encode(X_tensor).cpu().numpy()


        # Store just the latents in a dict
        latent_vars = {
            'train_latent': train_latent,
            'test_latent':  test_latent
        }

        # Save results
        results_one_modality = {
            'ids': ids,
            'results': results,
            'all_true': all_true,
            'all_pred': all_pred,
            'latent_vars': latent_vars,
            'final_latent': discovery_latents
        }

        # Combine all modalities into a dictionary
        results_by_modality[modality] = results_one_modality

    return results_by_modality
