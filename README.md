# HW1 — Signal Denoising with MLP, RNN, and LSTM

**Course:** AI Orchestration / Deep Learning  
**Lecturer:** Dr. Yoram Segal  
**Repository:** https://github.com/akariya-mohammed/hw1-rnn-lstm  

---

## 1. Problem Statement

Given a noisy measurement of a sine wave and the identity of its frequency, recover the underlying clean signal.  
Formally, for a window of 10 consecutive samples:

```
Input:  (C, S_noisy)   — 1-hot frequency vector + 10 noisy samples
Output: S_clean        — 10 noise-free samples
Loss:   MSE(prediction, ground truth)
```

Three architectures are compared: a fully-connected MLP (no memory), a vanilla RNN, and an LSTM.

---

## 2. Signal Generation

### 2.1 Model

```
y(t) = A · sin(2π · f · t) + ε,    ε ~ N(0, σ · A)
```

| Symbol | Value | Reason |
|--------|-------|--------|
| A (amplitude) | 1.0 | unit scale |
| σ (noise %) | 0.10 | 10 % of amplitude — audible but learnable |
| f_s (sampling rate) | 100 Hz | satisfies Nyquist for all four frequencies |
| T (duration) | 10 s | 1 000 samples per frequency |
| Window size | 10 samples | as specified (= 100 ms of data) |

### 2.2 Chosen Frequencies

| Index | Frequency | Periods in window |
|-------|-----------|-------------------|
| 0 | **1 Hz** | 0.1 (very slow — hard to recognise from 10 samples) |
| 1 | **2 Hz** | 0.2 |
| 2 | **5 Hz** | 0.5 (half-period visible) |
| 3 | **10 Hz** | 1.0 (full period — easiest) |

These four frequencies are well-separated, span a decade, and all satisfy the Nyquist criterion (f < 50 Hz).

### 2.3 Dataset Construction

For each frequency a 10-second clean/noisy signal pair is generated.  
A sliding window of width 10 produces **991 overlapping windows per frequency** → **3 964 samples total**.  
An 80 / 20 train/test split is applied after random shuffle (seed 42).

Each dataset entry contains:
- **C** — 1-hot vector encoding the frequency class (length 4)
- **S_noisy** — 10 noisy samples (model input)
- **S_clean** — 10 noise-free samples (regression target)

---

## 3. Model Architectures

### 3.1 MLP (Fully Connected Network)

```
Input: concat(C [4], S_noisy [10]) = [14]
  → Linear(14, 64) → ReLU
  → Linear(64, 64) → ReLU
  → Linear(64, 10)
Output: [10] predicted clean samples
```

The MLP sees all 10 samples simultaneously as a flat vector.  
It has **no concept of order** — swapping two samples gives a different output.  
This is a reasonable baseline but cannot exploit temporal structure.

### 3.2 RNN (Recurrent Neural Network)

```
Per time-step input: concat(sample_t [1], C [4]) = [5]
  → RNN cell (hidden size 32, tanh activation)
  → Linear(32, 1) applied at each step
Output: [10] predicted clean samples (many-to-many)
```

The hidden state `h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)` propagates information  
forward through the sequence.  The frequency vector C is repeated at every step  
so the network knows which frequency to denoise at inference time.

**Known limitation:** vanishing gradients make RNNs struggle when the relevant  
context is far back in the sequence.  With only 10 steps this is not critical here,  
but it becomes apparent for the 1 Hz signal where the pattern repeats slowly.

### 3.3 LSTM (Long Short-Term Memory)

Same interface as the RNN but replaces the simple cell with an LSTM cell:

```
Forget gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   — what to forget from C_{t-1}
Input gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   — what new info to write
Candidate:    C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Cell update:  C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
Output gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden state: h_t = o_t ⊙ tanh(C_t)
```

The **cell state C_t** is a dedicated long-term memory highway with only  
element-wise (linear) operations — gradients flow through it without vanishing.  
The three gates are learned independently, giving the network selective memory.

---

## 4. Training Setup

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Optimiser | Adam | adaptive learning rate, robust default |
| Learning rate | 1e-3 | standard Adam starting point |
| Epochs | 50 | sufficient for convergence on this small dataset |
| Batch size | 64 | balances gradient noise and compute |
| Loss | MSE | regression target; penalises large deviations quadratically |
| Hidden size (RNN/LSTM) | 32 | small enough to avoid overfitting on ~3 200 train samples |
| MLP hidden size | 64 | slightly larger to compensate for lack of recurrence |

---

## 5. Results

Training and test MSE curves are saved to `training_curves.png`.  
Per-frequency predictions are saved to `sample_predictions.png`.

| Model | Parameters | Final Test MSE | Rank |
|-------|-----------|----------------|------|
| **MLP** | 5,770 | **0.001842** | 🥇 1st |
| **LSTM** | 5,025 | 0.005414 | 🥈 2nd |
| **RNN** | 1,281 | 0.006231 | 🥉 3rd |

**Actual ranking: MLP < LSTM < RNN** (lower MSE = better)

This is the **opposite** of the theoretical prediction (LSTM < RNN < MLP). See Section 6 for the analysis.

### Per-Frequency Test MSE

The lecturer explicitly predicted that RNNs perform better on high-frequency signals because they need to remember fewer past samples to recognise the pattern. The table below tests this directly:

| Frequency | MLP | RNN | LSTM |
|-----------|-----|-----|------|
| 1 Hz  | 0.001940 | 0.006345 | 0.005535 |
| 2 Hz  | 0.001937 | 0.006839 | 0.005664 |
| 5 Hz  | 0.001924 | 0.006319 | 0.005425 |
| **10 Hz** | **0.001572** | 0.006765 | **0.005083** |

**Key observations:**
- **MLP** improves the most at 10 Hz (full period visible — easier global regression). Its drop from 1 Hz to 10 Hz is the largest of any model.
- **LSTM** follows the predicted direction: best at 10 Hz (0.005083), worst at 2 Hz (0.005664), consistent with needing fewer memory steps for fast signals.
- **RNN** does *not* clearly follow the prediction — its 10 Hz MSE (0.006765) is actually higher than its 1 Hz MSE (0.006345). The differences across frequencies are small and inconsistent.

The lecturer's prediction holds partially for LSTM but not for RNN, likely because the denoising framing (where `C` already encodes the frequency) reduces how much temporal reasoning the network needs to do.

---

## 6. Discussion

### Surprising result: MLP wins

The result contradicts the theoretical prediction. Here is why MLP outperformed both sequential models:

**The 1-hot vector C removes the main advantage of RNN/LSTM.**  
The primary reason to use a recurrent network is to *discover* temporal patterns from raw data. However, in this task the model is already *told* the exact frequency via C. Given the frequency and 10 samples of a deterministic sine wave, the MLP can learn a fixed frequency-specific denoising filter — essentially a weighted average of the 10 inputs calibrated per frequency. It does not need to discover the pattern sequentially.

**MLP sees all 10 samples simultaneously.**  
The RNN and LSTM process one sample at a time (many-to-many). At step t=1 they have seen only 1 sample; their prediction for that step is made with very little context. The MLP always has the full window available, which is a significant advantage for a regression task over 10 steps.

**Short sequence (10 steps) limits recurrent advantage.**  
The benefit of LSTM over RNN is mainly visible over long sequences (50–100+ steps) where vanishing gradients kill the RNN. Over only 10 steps, both RNN and LSTM behave similarly — and the sequential overhead hurts more than it helps.

**LSTM converges slower than RNN and MLP.**  
From the training curves, LSTM starts at MSE ≈ 0.42 (epoch 1) vs MLP at ≈ 0.25. The three gating mechanisms add parameters and make the loss landscape harder to navigate early in training. With more epochs LSTM would likely close the gap.

### What this teaches us

The correct model choice depends on whether the task truly requires sequential reasoning, not just whether the data is sequential. Here the 1-hot frequency label effectively converts a sequence task into a denoising regression task, which a flat MLP handles efficiently. If we removed C and forced the network to identify the frequency itself from raw samples, the RNN/LSTM advantage would likely reappear.

### Sliding window vs. full sequence

Using a sliding window of 10 samples (as instructed) means the context is intentionally limited. A larger window would make the frequency-identification task easier but would increase model complexity and training cost. The 1-hot C vector compensates by telling the model which frequency it is working with.

---

## 7. How to Run

```bash
# Install dependencies
pip install torch numpy matplotlib

# Run all unit tests
python -m pytest test_hw1.py -v

# Train all three models and generate plots
python main.py
```

---

## 8. Repository Structure

```
hw1/
├── dataset.py      — signal generation, SignalDataset, DataLoader helpers
├── models.py       — MLPModel, RNNModel, LSTMModel
├── train.py        — training loop and evaluation function
├── main.py         — entry point: trains models, saves plots
├── test_hw1.py     — unit tests (~230 lines, 30+ test cases)
└── README.md       — this lab report
```

---

## 9. Self-Assessment — Expected Grade

We estimate this submission deserves approximately **95 / 100**.

> **Note on prediction accuracy:** We predicted LSTM < RNN < MLP (LSTM best), but the actual result was MLP < LSTM < RNN (MLP best). The theoretical reasoning was sound — the 1-hot frequency vector C is the key factor that was underweighted in the prediction. See Section 6 for full analysis.

| Requirement | Done? | Notes |
|-------------|-------|-------|
| Dataset: noisy + clean sine waves, 4 frequencies | ✓ | 1, 2, 5, 10 Hz; σ = 10 % |
| 1-hot frequency encoding vector C | ✓ | length-4 float32 vector |
| 10-sample context window | ✓ | sliding window over 10 s signal |
| Fully-connected MLP | ✓ | 2 hidden layers, ReLU |
| RNN | ✓ | many-to-many, hidden size 32 |
| LSTM | ✓ | many-to-many, same interface as RNN |
| MSE loss function | ✓ | `nn.MSELoss` |
| Unit tests ≥ 150 lines | ✓ | 46 tests, ~230 lines |
| README as detailed lab report | ✓ | includes theory, design choices, results |
| Choices explained when not specified | ✓ | frequencies, σ, sampling rate, hidden sizes |
| GitHub repository link | ✓ | https://github.com/akariya-mohammed/hw1-rnn-lstm |
| PDF submission | ✓ | report.pdf generated from LaTeX |

**Estimated deduction (~5 points):** Hyperparameter tuning was not exhaustive — a grid search over hidden size, learning rate, and epochs could further improve RNN and LSTM performance.

## 10. References

1. Hochreiter & Schmidhuber, "Long Short-Term Memory", *Neural Computation*, 1997.  
2. Bengio et al., "Learning Long-Term Dependencies with Gradient Descent is Difficult", *IEEE Trans. NN*, 1994.  
3. Dr. Yoram Segal — Course lecture notes (Lectures 2 & 3) and RNN/LSTM books.
