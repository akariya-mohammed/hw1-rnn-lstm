# TODO — Task List
## HW1: Signal Denoising with Neural Networks

**Group:** uoh-rl07  
**Members:** Mohammad Akariya (211862024) · Jude Khleif (325233633)  

---

## Completed Tasks

### Data Pipeline
- [x] `generate_signal(freq, noise_pct, rng)` — clean and noisy sine wave arrays
- [x] `freq_to_onehot(freq_idx)` — 1-hot frequency encoding
- [x] `SignalDataset` — sliding window dataset, float32 tensors
- [x] `get_dataloaders()` — 80/20 split with fixed seed
- [x] `generate_combined_signal()` — superposition of all four frequencies
- [x] `CombinedSignalDataset` — combined input, single-component target
- [x] `get_combined_dataloaders()` — dataloaders for supplementary task

### Models
- [x] `MLPModel` — Linear(14,32) → ReLU → Linear(32,32) → ReLU → Linear(32,10)
- [x] `RNNModel` — vanilla RNN, many-to-many, hidden size 32
- [x] `LSTMModel` — LSTM, many-to-many, hidden size 32
- [x] `BiRNNModel` — bidirectional, 2 layers, hidden 32, dropout 0.1
- [x] `BiLSTMModel` — bidirectional, 2 layers, hidden 64, dropout 0.2

### Training
- [x] `train_model()` — Adam, MSE loss, per-epoch logging
- [x] `evaluate_model()` — overall test MSE
- [x] `evaluate_per_frequency()` — per-frequency MSE breakdown
- [x] LR scheduler — ReduceLROnPlateau (factor 0.5, patience 10)
- [x] Gradient clipping — max norm 1.0

### Main Script
- [x] Train all five models for 200 epochs
- [x] Print overall and per-frequency results
- [x] Save `training_curves.png` and `sample_predictions.png`
- [x] `combined_experiment()` — full pipeline for supplementary task
- [x] Save `combined_training_curves.png` and `combined_sample_predictions.png`

### Testing (74 tests total)
- [x] `TestGenerateSignal` — shape, dtype, noise level, reproducibility (8 tests)
- [x] `TestFreqToOnehot` — shape, dtype, sum, correct index (5 tests)
- [x] `TestSignalDataset` — length, shapes, dtypes, reproducibility (10 tests)
- [x] `TestGetDataloaders` — split ratio, batch shapes (4 tests)
- [x] `TestMLPModel` — shape, finite, gradients, parameter count (5 tests)
- [x] `TestRNNModel` — shape, finite, gradients, order sensitivity (5 tests)
- [x] `TestLSTMModel` — shape, finite, cell state, differs from RNN (5 tests)
- [x] `TestBiRNNModel` — shape, finite, bidirectional flag, 2 layers (6 tests)
- [x] `TestBiLSTMModel` — shape, finite, bidirectional flag, 2 layers (7 tests)
- [x] `TestGenerateCombinedSignal` — shapes, zero-noise sum, reproducibility (6 tests)
- [x] `TestCombinedSignalDataset` — length, shapes, shared input/diff targets (10 tests)
- [x] `TestMSELoss` — perfect prediction, non-negative, formula (3 tests)

### Documentation
- [x] README.md — full lab report with all sections
- [x] report.tex / report.pdf — LaTeX report, 10 pages
- [x] docs/PRD.md — product requirements
- [x] docs/PLAN.md — implementation plan
- [x] docs/TODO.md — this file
- [x] requirements.txt
- [x] .gitignore

### Repository
- [x] GitHub repo created: github.com/akariya-mohammed/hw1-rnn-lstm
- [x] README at repo root
- [ ] Share repo with rmisegal@gmail.com

---

## Results Summary

| Model | Test MSE | Rank |
|-------|----------|------|
| BiLSTM | 0.001078 | 1st |
| BiRNN  | 0.001458 | 2nd |
| MLP    | 0.001580 | 3rd |
| LSTM   | 0.004681 | 4th |
| RNN    | 0.004944 | 5th |

**Expected grade: 95 / 100**
