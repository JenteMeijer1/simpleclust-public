"""Train and evaluate variational autoencoders for modality reduction."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import dill
import os
from joblib import Parallel as _Parallel, delayed as _delayed
import math

import torch.nn.functional as F
import random


from skopt import gp_minimize
from skopt.space import Categorical
from skopt.callbacks import DeltaYStopper



# ─── Model definition ─────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim,
                 dropout_rate=0.1, activation_fn=nn.ReLU()):
        """Initialize the object."""
        super().__init__()
        # Expanded to three hidden layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Mirror architecture in decoder (three hidden layers, no dropout)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, input_dim),
        )

    def reparameterize(self, mu, logvar):
        # Prevent numerical overflow by clamping logvar
        """Handle reparameterize."""
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu, logvar):
        """
        Compute the KL divergence between N(mu, sigma^2) and standard normal N(0,1).
        """
        # Clamp for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        # KL per sample: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_per_dim = 1 + logvar - mu.pow(2) - logvar.exp()
        kl = -0.5 * kl_per_dim.sum(dim=1)
        return kl.mean()

    def forward(self, x):
        """Handle forward."""
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


    def encode(self, x):
        """Handle encode."""
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu



# Deterministic decoding for evaluation: use z = mu (no sampling)
def _deterministic_decode(model, xb):
    """Handle deterministic decode."""
    hidden = model.encoder(xb)
    mu = model.fc_mu(hidden)
    recon = model.decoder(mu)
    return recon

# ─── Training & evaluation helpers ──────────────────────────────────────────
def train_model(
    model,
    dataloader,
    num_epochs,
    lr,
    device,
    warmup_epochs=100,
    ramp_epochs=50,
    beta_max=0.05,
    collect_history=False,
    early_stop_patience=0,
    warmup_patience=10,
    warmup_delta=1e-3,
    l1_reg=0.0
):


    """Train model."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    # Warmup plateau detection
    orig_warmup = warmup_epochs
    best_warm_recon = float('inf')
    warm_no_improve = 0
    # Early stopping setup
    best_total_loss = float('inf')
    epochs_no_improve = 0
    if collect_history:
        # Pre-allocate lists for speed
        recon_history = [0.0] * num_epochs
        kl_history    = [0.0] * num_epochs
    for epoch in range(1, num_epochs+1):
        # Determine β for this epoch (unchanged)
        if epoch <= warmup_epochs:
            beta = 0.0
        elif epoch <= warmup_epochs + ramp_epochs:
            frac = (epoch - warmup_epochs) / ramp_epochs
            beta = frac * beta_max
        else:
            beta = beta_max

        recon_epoch_loss = 0.0
        kl_epoch_loss = 0.0

        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            recon, mu, logvar = model(xb)
            recon_loss = criterion(recon, yb)
            kl_loss    = model.kl_divergence(mu, logvar)
            l1_loss = float(l1_reg) * mu.abs().mean() if l1_reg else recon_loss.new_tensor(0.0)
            loss       = recon_loss + beta * kl_loss + l1_loss

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            recon_epoch_loss += recon_loss.item()
            kl_epoch_loss += kl_loss.item()

        avg_recon = recon_epoch_loss / len(dataloader)
        avg_kl = kl_epoch_loss / len(dataloader)

        # Warmup plateau early-exit into ramp
        if warmup_patience > 0 and epoch <= orig_warmup:
            # Only treat as improvement if recon decreases by at least warmup_delta
            if avg_recon < best_warm_recon - warmup_delta:
                best_warm_recon = avg_recon
                warm_no_improve = 0
            else:
                warm_no_improve += 1
            if warm_no_improve >= warmup_patience:
                # shorten warmup to current epoch -> enter ramp next epoch
                warmup_epochs = epoch

        # Early stopping check once at full β
        if early_stop_patience > 0 and epoch > warmup_epochs + ramp_epochs:
            total_loss = avg_recon + beta * avg_kl
            if total_loss < best_total_loss:
                best_total_loss = total_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                # stop training early
                final_epoch = epoch
                break
        # Optionally log per-epoch stats without a progress bar
        # e.g. print(f"Epoch {epoch}: recon={avg_recon:.4f}, kl={avg_kl:.4f}, beta={beta:.4f}")
        if collect_history:
            recon_history[epoch-1] = avg_recon
            kl_history[epoch-1]    = avg_kl
    # If early stopped, truncate history lists
    if collect_history and early_stop_patience > 0 and 'final_epoch' in locals():
        recon_history = recon_history[:final_epoch]
        kl_history = kl_history[:final_epoch]
    if collect_history:
        return recon_history, kl_history


#
# Evaluation is deterministic: z=mu (no sampling) to ensure stable metrics and embeddings.
def evaluate_model(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            # Deterministic evaluation: z = mu, no sampling noise
            recon = _deterministic_decode(model, xb)
            preds.append(recon.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    # Debug: check for NaNs in predictions or targets
    if np.isnan(preds).any() or np.isnan(trues).any():
        print(f"NaNs in evaluate_model: preds NaN={np.isnan(preds).any()}, trues NaN={np.isnan(trues).any()}")
    return mean_squared_error(trues, preds), preds


# ─── β‑VAE loss evaluation for validation ──────────────────────────────
def evaluate_beta_vae_loss(model, dataloader, device, beta=0.05, stochastic=False, l1_reg=0.0):
    """
    Validation metric aligned with β‑VAE training objective.
    Returns mean (MSE + beta * KL) over all samples in the dataloader.

    If stochastic=True, uses one reparameterized sample per input (standard ELBO estimate).
    If stochastic=False, decodes deterministically from mu while computing exact KL from mu/logvar.
    """
    model.eval()
    mse_sum = 0.0
    kl_sum  = 0.0
    l1_sum = 0.0
    n       = 0
    mse = nn.MSELoss(reduction='sum')
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            if stochastic:
                recon, mu, logvar = model(xb)  # samples z
            else:
                hidden = model.encoder(xb)
                mu     = model.fc_mu(hidden)
                logvar = model.fc_logvar(hidden)
                recon  = model.decoder(mu)
            # reconstruction term
            mse_sum += mse(recon, yb).item()
            # KL term (exact from mu, logvar)
            logvar = torch.clamp(logvar, min=-10, max=10)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_sum += kl.sum().item()
            if l1_reg:
                l1_sum += float(l1_reg) * mu.abs().mean(dim=1).sum().item()
            n += xb.size(0)
    return (mse_sum + beta * kl_sum + l1_sum) / n

# ─── Helper class for early stopping in bayesian optimisation ─────────────────────────────────────────
class MinCallsDeltaStopper:
    """
    Only starts checking for no‐improvement after a minimum number of calls.
    """
    def __init__(self, delta, n_best, min_calls):
        """Initialize the object."""
        self._stopper = DeltaYStopper(delta=delta, n_best=n_best)
        self._min_calls = min_calls

    def __call__(self, res):
        # Skip early‐stop until we've done the initial calls
        """Handle call  ."""
        if len(res.func_vals) < self._min_calls:
            return False
        return self._stopper(res)


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


    # Use all logical CPUs on this node
    """Run nested cv autoencoder."""
    total_cores = os.cpu_count() or 1
    outer_jobs = min(outer_splits, total_cores)
    inner_jobs = max(1, total_cores // outer_jobs)

    outer_kf = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    results = []
    all_true, all_pred = [], []
    all_train_true, all_train_pred = [], []
    all_test_latents, all_train_latents = [], []
    final_recon_histories = []
    final_kl_histories = []

    # Helper function for processing a single CV fold in parallel
    def _process_fold(fold, train_idx, test_idx):
        """Process fold."""
        X_train, X_test = X[train_idx], X[test_idx]
        input_dim = X.shape[1]

        latent_dims_filtered = [d for d in latent_dims if d < input_dim]
        if not latent_dims_filtered:
            latent_dims_filtered = [max(1, input_dim - 1)]

        # Build search space over names, so we can record the activation key
        space = [
            Categorical(hidden_dims, name='hidden_dim'),
            Categorical(latent_dims_filtered, name='latent_dim'),
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
                    activation_fn=act_fn
                )
                tr_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(X_tr))
                tr_ld = DataLoader(tr_ds, batch_size=int(bs), shuffle=True, num_workers=0)
                train_model(
                    mdl, tr_ld,
                    num_epochs=200,
                    lr=lr,
                    device=device,
                    warmup_epochs=100,
                    ramp_epochs=50,
                    beta_max=0.05,
                    l1_reg=l1_reg
                )
                val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(X_val))
                val_ld = DataLoader(val_ds, batch_size=int(bs), shuffle=False, num_workers=0)
                val_loss = evaluate_beta_vae_loss(mdl, val_ld, device, beta=0.05, stochastic=False, l1_reg=l1_reg)
                return float(val_loss)

            # Parallel inner‐CV across splits
            losses = _Parallel(n_jobs=inner_jobs)(
                _delayed(_inner_eval)(i_tr, i_val)
                for i_tr, i_val in inner_kf.split(X_train)
            )
            return float(np.mean(losses))

        # GP-based Bayesian optimization
        # Early-stopping for Bayesian search: stop if best MSE doesn’t improve by >1e-3 over 10 iterations. Max 30 iterations.
        stopper = MinCallsDeltaStopper(delta=1e-3, n_best=1, min_calls=15)
        res = gp_minimize(objective, space, n_calls=20, random_state=42, n_initial_points = 15, callback=[stopper])

        best_h, best_l, best_act_name, best_lr, best_bs = res.x

        # Train final model on full X_train
        final = Autoencoder(
            input_dim=input_dim,
            hidden_dim=best_h,
            latent_dim=best_l,
            dropout_rate=0.1,
            activation_fn=activation_functions[best_act_name]
        )
        full_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(X_train))
        full_ld = DataLoader(full_ds, batch_size=int(best_bs), shuffle=True)

        recon_hist, kl_hist = train_model(
            final, full_ld,
            num_epochs=150,
            lr=best_lr,
            device=device,
            warmup_epochs=100,
            ramp_epochs=50,
            beta_max=0.05,
            collect_history=True,
            early_stop_patience=10,
            warmup_patience=10,
            l1_reg=l1_reg
        )
        final_recon_histories.append(recon_hist)
        final_kl_histories.append(kl_hist)


        # Evaluate on train and test
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(X_train))
        test_ds  = TensorDataset(torch.FloatTensor(X_test),  torch.FloatTensor(X_test))
        train_ld = DataLoader(train_ds, batch_size=int(best_bs), shuffle=False)
        test_ld  = DataLoader(test_ds,  batch_size=int(best_bs), shuffle=False)
        train_loss, train_pred = evaluate_model(final, train_ld, device)
        test_loss,  test_pred  = evaluate_model(final, test_ld,  device)

        # Compute full VAE loss (recon + beta_max*KL)
        beta_max = 0.05
        recon_sum_train = 0.0
        kl_sum_train    = 0.0
        criterion_sum   = nn.MSELoss(reduction='sum')
        final.eval()
        with torch.no_grad():
            for xb, yb in train_ld:
                xb, yb = xb.to(device), yb.to(device)
                recon, mu, logvar = final(xb)
                recon_sum_train += criterion_sum(recon, yb).item()
                kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                kl_sum_train   += kl_per_sample.sum().item()
        total_train_loss = (recon_sum_train + beta_max * kl_sum_train) / len(train_ld.dataset)

        recon_sum_test = 0.0
        kl_sum_test    = 0.0
        with torch.no_grad():
            for xb, yb in test_ld:
                xb, yb = xb.to(device), yb.to(device)
                recon, mu, logvar = final(xb)
                recon_sum_test += criterion_sum(recon, yb).item()
                kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                kl_sum_test    += kl_per_sample.sum().item()
        total_test_loss = (recon_sum_test + beta_max * kl_sum_test) / len(test_ld.dataset)



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

    # Ensure latent arrays have the same number of columns before stacking
    max_test_dim = max(arr.shape[1] for arr in all_test_latents)
    padded_test_latents = [
        np.pad(arr, ((0, 0), (0, max_test_dim - arr.shape[1])), mode='constant')
        for arr in all_test_latents
    ]
    max_train_dim = max(arr.shape[1] for arr in all_train_latents)
    padded_train_latents = [
        np.pad(arr, ((0, 0), (0, max_train_dim - arr.shape[1])), mode='constant')
        for arr in all_train_latents
    ]

    return (
        results,
        np.vstack(all_true),
        np.vstack(all_pred),
        np.vstack(padded_test_latents),
        np.vstack(padded_train_latents),
    )


# ─── Wrapper to accept a DataFrame or array ───────────────────────────────────
def prepare_and_run_autoencoder(df, device='cpu', hidden_dims=[128, 256, 512], activation_functions = {'LeakyReLU': nn.LeakyReLU(), 'selu': nn.SELU(), 'swish': nn.SiLU()}, learning_rates=[0.001, 0.0001], batch_sizes=[32, 64, 128], latent_dims=[2, 5, 10, 20], l1_reg=0.0):
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
        print(f"▶️  Running variational autoencoder for modality: {modality}")

        ids, matrix = df

        # Ensure latent dimensions are undercomplete (< number of input features)
        latent_dims_list = [2,5,10,20]
        input_dim = matrix.shape[1]
        latent_dims_filtered = [d for d in latent_dims_list if d < input_dim]
        if not latent_dims_filtered:
            latent_dims_filtered = [max(1, input_dim - 1)]


        # Unpack the 7 outputs; we only need the 1st and last two here
        results, all_true, all_pred, test_latent, train_latent = \
            prepare_and_run_autoencoder(matrix, device=device,
                                        hidden_dims=[100, 250, 500, 1000],
                                        activation_functions = {'LeakyReLU': nn.LeakyReLU(), 'selu': nn.SELU(), 'swish': nn.SiLU()},
                                        learning_rates=[0.001, 0.0001],
                                        batch_sizes=[32, 64, 128],
                                        latent_dims=latent_dims_filtered)


        # Select best hyperparameters (minimum test_loss)
        best = min(results, key=lambda r: r['test_loss'])['params']

        X = matrix.values if isinstance(df, pd.DataFrame) else matrix

        # Initialize final model
        final = Autoencoder(
            input_dim=X.shape[1],
            hidden_dim=best['hidden_dim'],
            latent_dim=best['latent_dim'],
            activation_fn=activation_functions[best['activation']],
            dropout_rate=0.1
        )

        # Train on full data set
        full_ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(X))
        full_ld = DataLoader(full_ds, batch_size=int(best['batch_size']), shuffle=True)


        train_model(
            final,
            full_ld,
            num_epochs=150,
            lr=best['learning_rate'],
            device=device
        )

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




def load_and_encode(model_path, X_new, device="cpu", activation_functions = {'LeakyReLU': nn.LeakyReLU(), 'selu': nn.SELU(), 'swish': nn.SiLU()}):
    # Load full checkpoint (not just weights)
    """Load and encode."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Reconstruct the same architecture
    model = Autoencoder(
        input_dim = X_new.shape[1],
        hidden_dim = ckpt["hidden_dim"],
        latent_dim = ckpt["latent_dim"],
        activation_fn = activation_functions[ckpt["activation_name"]],
        dropout_rate = 0.1
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
                                        latent_dims=[2, 5, 10,20]):
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
        print(f"▶️  Running variational autoencoder for modality: {modality}")

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







def run_VAE_complete (modalities_dict, device='cpu', split = 'discovery', hidden_dims = [128, 256, 512], activation_functions = {'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(), 'selu': nn.SELU(), 'swish': nn.SiLU()}, learning_rates = [0.001, 0.0001], batch_sizes = [32, 64, 128], latent_dims = [2,5,10], l1_reg=0.0):
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
        print(f"▶️  Running variational autoencoder for modality: {modality}")
        ids, matrix = df
        # Verify matrix rows align with provided IDs
        if matrix.shape[0] != len(ids):
            print(f"[VAE] Warning: modality {modality} matrix rows {matrix.shape[0]} != ids {len(ids)}")

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
            dropout_rate=0.1
        )

        # Train on full data set
        full_ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(X))
        full_ld = DataLoader(full_ds, batch_size=int(best['batch_size']), shuffle=True)
        train_model(final, full_ld,
                    num_epochs=150,
                    lr=best['learning_rate'],
                    device=device,
                    l1_reg=l1_reg)



        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            discovery_latents = final.encode(X_tensor).cpu().numpy()
            # Check final_latent alignment and sample output
            if discovery_latents.shape[0] != len(ids):
                print(f"[VAE] Warning: final_latent rows {discovery_latents.shape[0]} != ids {len(ids)} for modality {modality}")


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
