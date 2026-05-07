import torch
import torch.nn as nn


def train_model(model, train_loader, test_loader, n_epochs=50, lr=1e-3, device="cpu"):
    """Train a model and return (train_losses, test_losses) per epoch.

    Techniques applied:
      - Adam optimiser
      - ReduceLROnPlateau: halves LR when test loss stops improving for 10 epochs
      - Gradient clipping (max norm 1.0): prevents exploding gradients in RNNs
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    train_losses, test_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        running = 0.0
        for freq_vec, noisy, clean in train_loader:
            freq_vec = freq_vec.to(device)
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            pred = model(freq_vec, noisy)
            loss = criterion(pred, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item()

        train_losses.append(running / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for freq_vec, noisy, clean in test_loader:
                freq_vec = freq_vec.to(device)
                noisy = noisy.to(device)
                clean = clean.to(device)
                pred = model(freq_vec, noisy)
                val_loss += criterion(pred, clean).item()
        test_losses.append(val_loss / len(test_loader))

        scheduler.step(test_losses[-1])

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1:3d}/{n_epochs} | "
                f"Train MSE: {train_losses[-1]:.6f} | "
                f"Test  MSE: {test_losses[-1]:.6f} | "
                f"LR: {current_lr:.2e}"
            )

    return train_losses, test_losses


def evaluate_model(model, test_loader, device="cpu"):
    """Return mean MSE over the test set."""
    model.eval()
    criterion = nn.MSELoss()
    total = 0.0
    with torch.no_grad():
        for freq_vec, noisy, clean in test_loader:
            freq_vec = freq_vec.to(device)
            noisy = noisy.to(device)
            clean = clean.to(device)
            total += criterion(model(freq_vec, noisy), clean).item()
    return total / len(test_loader)


def evaluate_per_frequency(model, test_loader, frequencies, device="cpu"):
    """Return {freq: mse} for each frequency separately."""
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    sums = {f: 0.0 for f in frequencies}
    counts = {f: 0 for f in frequencies}

    with torch.no_grad():
        for freq_vec, noisy, clean in test_loader:
            freq_vec = freq_vec.to(device)
            noisy = noisy.to(device)
            clean = clean.to(device)
            pred = model(freq_vec, noisy)

            freq_idx = freq_vec.argmax(dim=1)
            for i, fidx in enumerate(freq_idx):
                f = frequencies[fidx.item()]
                sums[f] += criterion(pred[i], clean[i]).item()
                counts[f] += clean.shape[1]

    return {f: sums[f] / counts[f] for f in frequencies}
