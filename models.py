import torch
import torch.nn as nn

from dataset import N_FREQS, WINDOW_SIZE


class MLPModel(nn.Module):
    """Fully-connected denoiser.

    Treats all 10 noisy samples as a flat vector and maps them
    (together with the frequency one-hot) to 10 clean predictions.
    No temporal structure is exploited.

    Input  : concat(freq_vec [N_FREQS], noisy_samples [WINDOW_SIZE])
    Output : clean_samples [WINDOW_SIZE]
    """

    def __init__(self, window_size=WINDOW_SIZE, n_freqs=N_FREQS, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window_size + n_freqs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, window_size),
        )

    def forward(self, freq_vec, noisy_samples):
        x = torch.cat([freq_vec, noisy_samples], dim=-1)
        return self.net(x)


class RNNModel(nn.Module):
    """Vanilla RNN denoiser (many-to-many).

    Processes noisy samples one step at a time.  At each step the input
    is [sample_t || freq_vec], and the output is the predicted clean value
    for that step.  The frequency vector provides the conditioning signal
    that lets the network adapt to each of the four frequencies.

    Input per step : [1 + N_FREQS]
    Output         : clean_samples [WINDOW_SIZE]
    """

    def __init__(self, n_freqs=N_FREQS, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size=1 + n_freqs, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, freq_vec, noisy_samples):
        B, T = noisy_samples.shape
        freq_exp = freq_vec.unsqueeze(1).expand(-1, T, -1)          # (B, T, N_FREQS)
        x = torch.cat([noisy_samples.unsqueeze(-1), freq_exp], -1)  # (B, T, 1+N_FREQS)
        out, _ = self.rnn(x)                                         # (B, T, H)
        return self.fc(out).squeeze(-1)                              # (B, T)


class LSTMModel(nn.Module):
    """LSTM denoiser (many-to-many).

    Same interface as RNNModel but uses an LSTM cell, which carries an
    explicit cell state C_t in addition to the hidden state h_t.
    The cell state acts as a long-term memory highway, letting the network
    retain information across the 10-step window without gradient vanishing.

    Input per step : [1 + N_FREQS]
    Output         : clean_samples [WINDOW_SIZE]
    """

    def __init__(self, n_freqs=N_FREQS, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1 + n_freqs, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, freq_vec, noisy_samples):
        B, T = noisy_samples.shape
        freq_exp = freq_vec.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([noisy_samples.unsqueeze(-1), freq_exp], -1)
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)


class BiRNNModel(nn.Module):
    """Bidirectional stacked RNN denoiser (many-to-many).

    Processes the sequence in both directions across 2 stacked layers.
    Each time step sees context from past AND future samples, which is
    valid for denoising since the full window is available at inference.
    The two directions are concatenated, so the fc input is 2 * hidden_size.

    Input per step : [1 + N_FREQS]
    Output         : clean_samples [WINDOW_SIZE]
    """

    def __init__(self, n_freqs=N_FREQS, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1 + n_freqs, hidden_size=hidden_size,
            num_layers=2, bidirectional=True, batch_first=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, freq_vec, noisy_samples):
        B, T = noisy_samples.shape
        freq_exp = freq_vec.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([noisy_samples.unsqueeze(-1), freq_exp], -1)
        out, _ = self.rnn(x)          # (B, T, 2*H)
        return self.fc(out).squeeze(-1)


class BiLSTMModel(nn.Module):
    """Bidirectional stacked LSTM denoiser (many-to-many).

    Combines techniques from the lectures:
      - Bidirectional: forward pass sees future context, backward pass sees past
      - Stacked (2 layers): first layer learns low-level patterns, second learns
        higher-level temporal structure
      - Dropout (0.2) between layers: regularisation to improve generalisation
      - Hidden size 64: more capacity than the unidirectional models

    Input per step : [1 + N_FREQS]
    Output         : clean_samples [WINDOW_SIZE]
    """

    def __init__(self, n_freqs=N_FREQS, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1 + n_freqs, hidden_size=hidden_size,
            num_layers=2, bidirectional=True, batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, freq_vec, noisy_samples):
        B, T = noisy_samples.shape
        freq_exp = freq_vec.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([noisy_samples.unsqueeze(-1), freq_exp], -1)
        out, _ = self.lstm(x)          # (B, T, 2*H)
        return self.fc(out).squeeze(-1)
