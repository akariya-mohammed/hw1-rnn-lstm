# PRD — Product Requirements Document
## HW1: Signal Denoising with Neural Networks

**Group:** uoh-rl07  
**Members:** Mohammad Akariya (211862024) · Jude Khleif (325233633)  
**Course:** AI Orchestration / Deep Learning — Dr. Yoram Segal  

---

## 1. Problem Statement

Build and compare neural network architectures that recover a clean sine-wave signal from a noisy, windowed measurement. The model is given the frequency identity of the signal and must output the denoised samples.

---

## 2. Functional Requirements

### 2.1 Dataset
- Generate sine waves at four frequencies: 1 Hz, 2 Hz, 5 Hz, 10 Hz
- Sampling rate: 100 Hz, duration: 10 seconds per frequency
- Additive Gaussian noise: σ = 10% of amplitude
- Sliding window of 10 samples → 991 windows per frequency → 3,964 total samples
- 80/20 train/test split, random shuffle with seed 42
- Each sample: `(C, S_noisy, S_clean)` where C is a 1-hot frequency vector

### 2.2 Models
| Model | Architecture |
|-------|-------------|
| MLP | Fully-connected, 2 hidden layers, hidden size 32 |
| RNN | Vanilla recurrent, many-to-many, hidden size 32 |
| LSTM | Gated recurrent (forget/input/output gates), hidden size 32 |
| BiRNN | Bidirectional stacked RNN, 2 layers, hidden size 32, dropout 0.1 |
| BiLSTM | Bidirectional stacked LSTM, 2 layers, hidden size 64, dropout 0.2 |

### 2.3 Training
- Loss: Mean Squared Error (MSE)
- Optimiser: Adam, initial LR = 1e-3
- LR scheduler: ReduceLROnPlateau (factor 0.5, patience 10)
- Gradient clipping: max norm 1.0
- Epochs: 200 (main task), 200 (supplementary combined task)
- Batch size: 64

### 2.4 Supplementary experiment
- Combined signal: superposition of all four frequencies + noise
- Task: extract one frequency component indicated by C
- Validates theoretical prediction that LSTM outperforms RNN on harder tasks

---

## 3. Non-Functional Requirements

- Unit tests: ≥ 150 lines, covering dataset, models, and loss function
- All code reproducible (fixed seeds)
- README as detailed lab report with results, discussion, and design decisions
- PDF report with title page, sections, figures, and tables
- GitHub repository shared with lecturer

---

## 4. Deliverables

| Deliverable | Status |
|-------------|--------|
| dataset.py | Done |
| models.py | Done |
| train.py | Done |
| main.py | Done |
| test_hw1.py | Done (74 tests) |
| README.md | Done |
| report.pdf | Done (10 pages) |
| GitHub repo | Done |
