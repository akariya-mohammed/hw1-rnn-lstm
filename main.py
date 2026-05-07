"""HW1 main script: train MLP, RNN, and LSTM on the signal denoising task."""

import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset import (
    get_dataloaders, get_combined_dataloaders,
    generate_signal, generate_combined_signal,
    FREQUENCIES, SAMPLE_RATE, WINDOW_SIZE,
)
from models import MLPModel, RNNModel, LSTMModel
from train import train_model, evaluate_model, evaluate_per_frequency

N_EPOCHS = 50
N_EPOCHS_COMBINED = 200
BATCH_SIZE = 64
LR = 1e-3
NOISE_PCT = 0.1


def plot_training_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, res in results.items():
        axes[0].plot(res["train_losses"], label=name)
        axes[1].plot(res["test_losses"], label=name)
    for ax, title in zip(axes, ["Train MSE", "Test MSE"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved training_curves.png")


def plot_sample_predictions(models_dict, device):
    """Visualise clean / noisy / predicted signals for each frequency."""
    fig, axes = plt.subplots(len(FREQUENCIES), 3, figsize=(15, 3 * len(FREQUENCIES)))
    model_names = list(models_dict.keys())

    rng = np.random.default_rng(0)
    for row, freq in enumerate(FREQUENCIES):
        clean, noisy = generate_signal(freq, noise_pct=NOISE_PCT, rng=rng)
        window = slice(0, WINDOW_SIZE)
        x_noisy = torch.tensor(noisy[window]).unsqueeze(0)
        x_clean = clean[window]

        freq_idx = FREQUENCIES.index(freq)
        freq_vec = torch.zeros(1, len(FREQUENCIES))
        freq_vec[0, freq_idx] = 1.0

        t = np.arange(WINDOW_SIZE)

        for col, name in enumerate(model_names):
            model = models_dict[name].to(device)
            model.eval()
            with torch.no_grad():
                pred = model(freq_vec.to(device), x_noisy.to(device)).cpu().numpy()[0]

            ax = axes[row, col]
            ax.plot(t, x_clean, "g-", label="clean", linewidth=2)
            ax.plot(t, noisy[window], "r--", alpha=0.6, label="noisy")
            ax.plot(t, pred, "b-", label="predicted", linewidth=1.5)
            ax.set_title(f"{name} | f={freq} Hz")
            ax.set_xlabel("sample")
            ax.set_ylabel("amplitude")
            if row == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150)
    print("Saved sample_predictions.png")


def combined_experiment(device):
    """Train all three models on the combined signal extraction task and save plots."""
    print(f"\n{'='*50}")
    print("SUPPLEMENTARY: Combined Signal Extraction")
    print("Task: extract one frequency component from a noisy mixture of all 4 frequencies")
    print(f"{'='*50}")

    train_loader, test_loader = get_combined_dataloaders(batch_size=BATCH_SIZE, noise_pct=NOISE_PCT)
    print(f"Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}\n")

    model_classes = {"MLP": MLPModel, "RNN": RNNModel, "LSTM": LSTMModel}
    results, trained_models = {}, {}

    for name, ModelClass in model_classes.items():
        print(f"--- Combined: {name} ---")
        model = ModelClass()
        train_losses, test_losses = train_model(
            model, train_loader, test_loader, n_epochs=N_EPOCHS_COMBINED, lr=LR, device=device
        )
        final_mse = evaluate_model(model, test_loader, device=device)
        results[name] = {"train_losses": train_losses, "test_losses": test_losses, "final_mse": final_mse}
        trained_models[name] = model

    print(f"\nCombined — Final Test MSE:")
    for name, res in results.items():
        print(f"  {name:5s}: {res['final_mse']:.6f}")

    # per-frequency breakdown
    print(f"\n{'='*50}")
    print("Combined — Per-Frequency Test MSE:")
    header = f"  {'Freq':>8}" + "".join(f"  {n:>10}" for n in results)
    print(header)
    per_freq = {name: evaluate_per_frequency(trained_models[name], test_loader, FREQUENCIES, device)
                for name in results}
    for f in FREQUENCIES:
        row = f"  {f:>6.0f} Hz" + "".join(f"  {per_freq[n][f]:>10.6f}" for n in results)
        print(row)

    # training curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, res in results.items():
        axes[0].plot(res["train_losses"], label=name)
        axes[1].plot(res["test_losses"], label=name)
    for ax, title in zip(axes, ["Combined — Train MSE", "Combined — Test MSE"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("combined_training_curves.png", dpi=150)
    print("Saved combined_training_curves.png")
    plt.close("all")

    # sample predictions
    rng = np.random.default_rng(0)
    components, combined_noisy = generate_combined_signal(noise_pct=NOISE_PCT, rng=rng)
    window = slice(0, WINDOW_SIZE)
    t_ax = np.arange(WINDOW_SIZE)
    x_combined = torch.tensor(combined_noisy[window]).unsqueeze(0)

    fig, axes = plt.subplots(len(FREQUENCIES), 3, figsize=(15, 3 * len(FREQUENCIES)))
    model_names = list(trained_models.keys())
    for row, (freq_idx, freq) in enumerate(zip(range(len(FREQUENCIES)), FREQUENCIES)):
        freq_vec = torch.zeros(1, len(FREQUENCIES))
        freq_vec[0, freq_idx] = 1.0
        target = components[freq_idx][window]
        for col, name in enumerate(model_names):
            model = trained_models[name].to(device)
            model.eval()
            with torch.no_grad():
                pred = model(freq_vec.to(device), x_combined.to(device)).cpu().numpy()[0]
            ax = axes[row, col]
            ax.plot(t_ax, target, "g-", label="target component", linewidth=2)
            ax.plot(t_ax, combined_noisy[window], "r--", alpha=0.35, label="combined input")
            ax.plot(t_ax, pred, "b-", label="predicted", linewidth=1.5)
            ax.set_title(f"{name} | extract f={freq} Hz")
            ax.set_xlabel("sample")
            ax.set_ylabel("amplitude")
            if row == 0:
                ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("combined_sample_predictions.png", dpi=150)
    print("Saved combined_sample_predictions.png")
    plt.close("all")

    return results, per_freq


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Frequencies: {FREQUENCIES} Hz  |  Sampling rate: {SAMPLE_RATE} Hz  |  Window: {WINDOW_SIZE} samples")

    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, noise_pct=NOISE_PCT)
    print(f"Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}\n")

    model_classes = {
        "MLP": MLPModel,
        "RNN": RNNModel,
        "LSTM": LSTMModel,
    }

    results = {}
    trained_models = {}

    for name, ModelClass in model_classes.items():
        print(f"{'='*50}")
        print(f"Training {name}")
        model = ModelClass()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")
        train_losses, test_losses = train_model(
            model, train_loader, test_loader,
            n_epochs=N_EPOCHS, lr=LR, device=device
        )
        final_mse = evaluate_model(model, test_loader, device=device)
        results[name] = {"train_losses": train_losses, "test_losses": test_losses, "final_mse": final_mse}
        trained_models[name] = model

    print(f"\n{'='*50}")
    print("Final Test MSE (overall):")
    for name, res in results.items():
        print(f"  {name:5s}: {res['final_mse']:.6f}")

    print(f"\n{'='*50}")
    print("Per-Frequency Test MSE:")
    header = f"  {'Freq':>8}" + "".join(f"  {n:>10}" for n in results)
    print(header)
    per_freq = {}
    for name, model in trained_models.items():
        per_freq[name] = evaluate_per_frequency(model, test_loader, FREQUENCIES, device)
    for f in FREQUENCIES:
        row = f"  {f:>6.0f} Hz" + "".join(f"  {per_freq[n][f]:>10.6f}" for n in results)
        print(row)

    plot_training_curves(results)
    plot_sample_predictions(trained_models, device)
    plt.close("all")

    combined_experiment(device)


if __name__ == "__main__":
    main()
