import torch
import torch.nn as nn


def train_model(model, train_loader, test_loader, n_epochs=50, lr=1e-3, device="cpu"):
    """Train a model and return (train_losses, test_losses) per epoch."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{n_epochs} | "
                f"Train MSE: {train_losses[-1]:.6f} | "
                f"Test  MSE: {test_losses[-1]:.6f}"
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

            # identify which frequency each sample belongs to
            freq_idx = freq_vec.argmax(dim=1)
            for i, fidx in enumerate(freq_idx):
                f = frequencies[fidx.item()]
                sums[f] += criterion(pred[i], clean[i]).item()
                counts[f] += clean.shape[1]  # number of elements per sample

    return {f: sums[f] / counts[f] for f in frequencies}
