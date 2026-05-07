# PLAN — Implementation Plan
## HW1: Signal Denoising with Neural Networks

**Group:** uoh-rl07  
**Members:** Mohammad Akariya (211862024) · Jude Khleif (325233633)  

---

## 1. Architecture Overview

```
hw1/
├── dataset.py    — signal generation and dataset classes
├── models.py     — all five model architectures
├── train.py      — shared training loop with scheduler and grad clipping
├── main.py       — entry point: orchestrates training and plotting
├── test_hw1.py   — unit tests
├── docs/         — PRD, PLAN, TODO
└── README.md     — lab report
```

---

## 2. Implementation Phases

### Phase 1 — Data Pipeline
- Implement `generate_signal(freq, noise_pct)` using NumPy
- Implement `SignalDataset` with sliding window logic
- Implement `get_dataloaders()` with 80/20 split
- Prompt to agent: define signal model, frequencies, window size, dataset structure

### Phase 2 — Baseline Models
- Implement `MLPModel`: flat concat of C and noisy samples → 2 hidden layers → output
- Implement `RNNModel`: many-to-many, C repeated at each step, fc on hidden state
- Implement `LSTMModel`: same interface as RNN, replace cell with LSTM gates
- Prompt to agent: specify input/output shapes, hidden sizes, many-to-many requirement

### Phase 3 — Training Loop
- Implement `train_model()` with Adam, MSE loss, per-epoch logging
- Implement `evaluate_model()` and `evaluate_per_frequency()`
- Add `ReduceLROnPlateau` scheduler and gradient clipping (max norm 1.0)
- Prompt to agent: specify logging interval, scheduler patience and factor

### Phase 4 — Extended Models
- Implement `BiRNNModel`: `nn.RNN(bidirectional=True, num_layers=2)`
- Implement `BiLSTMModel`: `nn.LSTM(bidirectional=True, num_layers=2, hidden_size=64)`
- Update `main.py` to include all five models
- Prompt to agent: specify bidirectional concatenation, fc input = 2 * hidden_size

### Phase 5 — Supplementary Experiment
- Implement `generate_combined_signal()`: sum of all four frequencies + noise
- Implement `CombinedSignalDataset`: input = combined noisy, target = one component
- Implement `combined_experiment()` in main.py with 200-epoch training
- Prompt to agent: define task as frequency-selective filtering, same model interface

### Phase 6 — Testing
- Write unit tests for all dataset classes, all models, and MSE loss
- Cover: shapes, dtypes, reproducibility, gradient flow, bidirectional flags
- Target: ≥ 150 lines, ≥ 30 test cases

### Phase 7 — Documentation
- Write README as lab report: signal model, architectures, results, discussion
- Write LaTeX report compiled to PDF
- Add PRD, PLAN, TODO to docs/

---

## 3. Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Window size | 10 samples | As specified in instructions |
| Frequencies | 1, 2, 5, 10 Hz | Span a decade, all below Nyquist (50 Hz) |
| Noise level | σ = 0.10 (fixed) | No range specified; fixed keeps task reproducible |
| Single vs combined signal | Both | Single for main task; combined as supplementary |
| BiLSTM hidden size | 64 (vs 32 for others) | More capacity to exploit bidirectional structure |
| Epochs | 200 | BiLSTM needs more epochs to converge |
| LR scheduler | ReduceLROnPlateau | Automatic fine-tuning when loss plateaus |
| Grad clipping | max norm 1.0 | Prevents exploding gradients in RNNs |
