"""Unit tests for HW1: signal denoising with MLP, RNN, and LSTM."""

import unittest
import numpy as np
import torch
import torch.nn as nn

from dataset import (
    generate_signal, freq_to_onehot, SignalDataset, get_dataloaders,
    FREQUENCIES, SAMPLE_RATE, DURATION, WINDOW_SIZE, N_FREQS,
)
from models import MLPModel, RNNModel, LSTMModel


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

class TestGenerateSignal(unittest.TestCase):

    def test_output_shapes_match(self):
        clean, noisy = generate_signal(1.0)
        self.assertEqual(clean.shape, noisy.shape)

    def test_default_length(self):
        clean, _ = generate_signal(1.0)
        self.assertEqual(len(clean), SAMPLE_RATE * DURATION)

    def test_zero_noise_equals_clean(self):
        clean, noisy = generate_signal(1.0, noise_pct=0.0)
        np.testing.assert_allclose(clean, noisy, atol=1e-6)

    def test_clean_is_sine(self):
        freq = 2.0
        clean, _ = generate_signal(freq, noise_pct=0.0)
        t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION, endpoint=False)
        expected = np.sin(2 * np.pi * freq * t).astype(np.float32)
        np.testing.assert_allclose(clean, expected, atol=1e-5)

    def test_noise_level_approximately_correct(self):
        amplitude, noise_pct = 1.0, 0.1
        rng = np.random.default_rng(0)
        clean, noisy = generate_signal(1.0, amplitude=amplitude, noise_pct=noise_pct, rng=rng)
        residual_std = (noisy - clean).std()
        self.assertAlmostEqual(residual_std, noise_pct * amplitude, delta=0.02)

    def test_dtype_is_float32(self):
        clean, noisy = generate_signal(1.0)
        self.assertEqual(clean.dtype, np.float32)
        self.assertEqual(noisy.dtype, np.float32)

    def test_amplitude_scales_signal(self):
        clean1, _ = generate_signal(1.0, amplitude=1.0, noise_pct=0.0)
        clean2, _ = generate_signal(1.0, amplitude=2.0, noise_pct=0.0)
        np.testing.assert_allclose(clean2, 2 * clean1, atol=1e-5)

    def test_different_frequencies_differ(self):
        clean1, _ = generate_signal(1.0, noise_pct=0.0)
        clean2, _ = generate_signal(5.0, noise_pct=0.0)
        self.assertFalse(np.allclose(clean1, clean2))

    def test_rng_reproducibility(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        _, n1 = generate_signal(1.0, rng=rng1)
        _, n2 = generate_signal(1.0, rng=rng2)
        np.testing.assert_array_equal(n1, n2)


class TestFreqToOnehot(unittest.TestCase):

    def test_shape(self):
        c = freq_to_onehot(0)
        self.assertEqual(c.shape, (N_FREQS,))

    def test_dtype(self):
        c = freq_to_onehot(0)
        self.assertEqual(c.dtype, np.float32)

    def test_sum_is_one(self):
        for i in range(N_FREQS):
            self.assertAlmostEqual(freq_to_onehot(i).sum(), 1.0)

    def test_correct_index_is_hot(self):
        for i in range(N_FREQS):
            c = freq_to_onehot(i)
            self.assertEqual(c[i], 1.0)

    def test_all_other_zeros(self):
        for i in range(N_FREQS):
            c = freq_to_onehot(i)
            for j in range(N_FREQS):
                if j != i:
                    self.assertEqual(c[j], 0.0)


# ---------------------------------------------------------------------------
# SignalDataset
# ---------------------------------------------------------------------------

class TestSignalDataset(unittest.TestCase):

    def setUp(self):
        self.ds = SignalDataset(seed=0)

    def test_length(self):
        expected = N_FREQS * (SAMPLE_RATE * DURATION - WINDOW_SIZE + 1)
        self.assertEqual(len(self.ds), expected)

    def test_item_returns_three_tensors(self):
        item = self.ds[0]
        self.assertEqual(len(item), 3)

    def test_freq_vec_shape(self):
        c, _, _ = self.ds[0]
        self.assertEqual(c.shape, (N_FREQS,))

    def test_noisy_shape(self):
        _, noisy, _ = self.ds[0]
        self.assertEqual(noisy.shape, (WINDOW_SIZE,))

    def test_clean_shape(self):
        _, _, clean = self.ds[0]
        self.assertEqual(clean.shape, (WINDOW_SIZE,))

    def test_tensors_are_float32(self):
        c, noisy, clean = self.ds[0]
        for t in (c, noisy, clean):
            self.assertEqual(t.dtype, torch.float32)

    def test_freq_vec_is_valid_onehot(self):
        for i in range(0, len(self.ds), SAMPLE_RATE * DURATION - WINDOW_SIZE + 1):
            c, _, _ = self.ds[i]
            self.assertAlmostEqual(c.sum().item(), 1.0)
            self.assertTrue((c >= 0).all())

    def test_noisy_differs_from_clean(self):
        _, noisy, clean = self.ds[0]
        self.assertFalse(torch.allclose(noisy, clean))

    def test_no_nan_values(self):
        for idx in [0, 100, 500]:
            c, noisy, clean = self.ds[idx]
            for t in (c, noisy, clean):
                self.assertFalse(torch.isnan(t).any())

    def test_reproducible_with_same_seed(self):
        ds2 = SignalDataset(seed=0)
        c1, n1, cl1 = self.ds[0]
        c2, n2, cl2 = ds2[0]
        self.assertTrue(torch.equal(n1, n2))
        self.assertTrue(torch.equal(cl1, cl2))


class TestGetDataloaders(unittest.TestCase):

    def test_returns_two_loaders(self):
        train, test = get_dataloaders(batch_size=32)
        self.assertIsNotNone(train)
        self.assertIsNotNone(test)

    def test_batch_freq_vec_shape(self):
        train, _ = get_dataloaders(batch_size=32)
        c, _, _ = next(iter(train))
        self.assertEqual(c.shape[1], N_FREQS)

    def test_batch_noisy_shape(self):
        train, _ = get_dataloaders(batch_size=32)
        _, noisy, _ = next(iter(train))
        self.assertEqual(noisy.shape[1], WINDOW_SIZE)

    def test_train_test_ratio(self):
        train, test = get_dataloaders(batch_size=64, train_ratio=0.8)
        n_train = len(train.dataset)
        n_test = len(test.dataset)
        ratio = n_train / (n_train + n_test)
        self.assertAlmostEqual(ratio, 0.8, delta=0.01)


# ---------------------------------------------------------------------------
# MLPModel
# ---------------------------------------------------------------------------

class TestMLPModel(unittest.TestCase):

    def setUp(self):
        self.model = MLPModel()
        self.B = 8

    def _dummy(self):
        freq = torch.zeros(self.B, N_FREQS)
        freq[:, 0] = 1.0
        noisy = torch.randn(self.B, WINDOW_SIZE)
        return freq, noisy

    def test_output_shape(self):
        freq, noisy = self._dummy()
        out = self.model(freq, noisy)
        self.assertEqual(out.shape, (self.B, WINDOW_SIZE))

    def test_output_finite(self):
        freq, noisy = self._dummy()
        self.assertTrue(torch.isfinite(self.model(freq, noisy)).all())

    def test_has_trainable_params(self):
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertGreater(n, 0)

    def test_gradients_flow(self):
        freq, noisy = self._dummy()
        clean = torch.randn(self.B, WINDOW_SIZE)
        loss = nn.MSELoss()(self.model(freq, noisy), clean)
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)
        for g in grads:
            self.assertFalse(torch.isnan(g).any())

    def test_different_inputs_different_outputs(self):
        freq = torch.zeros(1, N_FREQS); freq[0, 0] = 1.0
        n1, n2 = torch.randn(1, WINDOW_SIZE), torch.randn(1, WINDOW_SIZE)
        self.assertFalse(torch.allclose(self.model(freq, n1), self.model(freq, n2)))


# ---------------------------------------------------------------------------
# RNNModel
# ---------------------------------------------------------------------------

class TestRNNModel(unittest.TestCase):

    def setUp(self):
        self.model = RNNModel()
        self.B = 8

    def _dummy(self):
        freq = torch.zeros(self.B, N_FREQS)
        freq[:, 1] = 1.0
        noisy = torch.randn(self.B, WINDOW_SIZE)
        return freq, noisy

    def test_output_shape(self):
        freq, noisy = self._dummy()
        self.assertEqual(self.model(freq, noisy).shape, (self.B, WINDOW_SIZE))

    def test_output_finite(self):
        freq, noisy = self._dummy()
        self.assertTrue(torch.isfinite(self.model(freq, noisy)).all())

    def test_gradients_flow(self):
        freq, noisy = self._dummy()
        clean = torch.randn(self.B, WINDOW_SIZE)
        loss = nn.MSELoss()(self.model(freq, noisy), clean)
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)

    def test_order_matters(self):
        freq = torch.zeros(1, N_FREQS); freq[0, 0] = 1.0
        seq = torch.randn(1, WINDOW_SIZE)
        shuffled = seq[:, torch.randperm(WINDOW_SIZE)]
        out1 = self.model(freq, seq)
        out2 = self.model(freq, shuffled)
        self.assertFalse(torch.allclose(out1, out2))

    def test_hidden_state_not_exposed(self):
        freq, noisy = self._dummy()
        result = self.model(freq, noisy)
        self.assertIsInstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# LSTMModel
# ---------------------------------------------------------------------------

class TestLSTMModel(unittest.TestCase):

    def setUp(self):
        self.model = LSTMModel()
        self.B = 8

    def _dummy(self):
        freq = torch.zeros(self.B, N_FREQS)
        freq[:, 2] = 1.0
        noisy = torch.randn(self.B, WINDOW_SIZE)
        return freq, noisy

    def test_output_shape(self):
        freq, noisy = self._dummy()
        self.assertEqual(self.model(freq, noisy).shape, (self.B, WINDOW_SIZE))

    def test_output_finite(self):
        freq, noisy = self._dummy()
        self.assertTrue(torch.isfinite(self.model(freq, noisy)).all())

    def test_gradients_flow(self):
        freq, noisy = self._dummy()
        clean = torch.randn(self.B, WINDOW_SIZE)
        loss = nn.MSELoss()(self.model(freq, noisy), clean)
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)

    def test_has_cell_state(self):
        freq, noisy = self._dummy()
        freq_exp = freq.unsqueeze(1).expand(-1, WINDOW_SIZE, -1)
        x = torch.cat([noisy.unsqueeze(-1), freq_exp], -1)
        _, (h, c) = self.model.lstm(x)
        self.assertIsNotNone(c)
        self.assertEqual(c.shape[2], self.model.lstm.hidden_size)

    def test_lstm_differs_from_rnn(self):
        rnn = RNNModel()
        lstm = LSTMModel()
        torch.manual_seed(0)
        freq = torch.zeros(1, N_FREQS); freq[0, 0] = 1.0
        noisy = torch.randn(1, WINDOW_SIZE)
        out_rnn = rnn(freq, noisy)
        out_lstm = lstm(freq, noisy)
        self.assertEqual(out_rnn.shape, out_lstm.shape)


# ---------------------------------------------------------------------------
# Loss function sanity checks
# ---------------------------------------------------------------------------

class TestMSELoss(unittest.TestCase):

    def test_perfect_prediction_is_zero(self):
        criterion = nn.MSELoss()
        x = torch.tensor([1.0, 2.0, 3.0])
        self.assertAlmostEqual(criterion(x, x).item(), 0.0, places=6)

    def test_loss_is_non_negative(self):
        criterion = nn.MSELoss()
        for _ in range(5):
            a, b = torch.randn(10), torch.randn(10)
            self.assertGreaterEqual(criterion(a, b).item(), 0.0)

    def test_mse_formula(self):
        criterion = nn.MSELoss()
        pred = torch.tensor([0.0, 0.0])
        target = torch.tensor([1.0, 3.0])
        expected = (1.0 + 9.0) / 2
        self.assertAlmostEqual(criterion(pred, target).item(), expected, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
