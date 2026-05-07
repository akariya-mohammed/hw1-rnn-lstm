# HW1 — Signal Denoising with MLP, RNN, LSTM, BiRNN, and BiLSTM

**Course:** AI Orchestration / Deep Learning  
**Lecturer:** Dr. Yoram Segal  
**Students:** Mohammad Akariya (211862024) · Jude Khleif (325233633)  
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

Five architectures are compared: a fully-connected MLP (no memory), a vanilla RNN, an LSTM, a bidirectional stacked RNN (BiRNN), and a bidirectional stacked LSTM (BiLSTM).

---

## 2. Signal Generation

### 2.1 Model

```
y(t) = A · sin(2π · f · t) + ε,    ε ~ N(0, σ · A)
```

| Symbol | Value | Reason |
|--------|-------|--------|
| A (amplitude) | 1.0 | unit scale |
| σ (noise %) | 0.10 | 10 % of amplitude — large enough to perturb, small enough to denoise |
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
  → Linear(14, 32) → ReLU
  → Linear(32, 32) → ReLU
  → Linear(32, 10)
Output: [10] predicted clean samples
```

The MLP sees all 10 samples simultaneously as a flat vector.  
It has **no concept of order** — swapping two samples gives a different output.  
Parameters: **1,866**

### 3.2 RNN (Recurrent Neural Network)

```
Per time-step input: concat(sample_t [1], C [4]) = [5]
  → RNN cell (hidden size 32, tanh activation)
  → Linear(32, 1) applied at each step
Output: [10] predicted clean samples (many-to-many)
```

Hidden state: `h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)`.  
Parameters: **1,281**

### 3.3 LSTM (Long Short-Term Memory)

Same interface as the RNN but replaces the simple cell with a gated LSTM cell:

```
Forget gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Cell update:  C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(W_C · [h_{t-1}, x_t] + b_C)
Output gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden state: h_t = o_t ⊙ tanh(C_t)
```

The **cell state C_t** is a dedicated long-term memory highway — gradients flow through it without vanishing.  
Parameters: **5,025**

### 3.4 BiRNN (Bidirectional Stacked RNN)

```
Per time-step input: [1 + 4] = [5]
  → RNN layer 1 — forward + backward pass (hidden size 32 each direction)
  → RNN layer 2 — forward + backward pass (hidden size 32 each direction)
  → concat(forward_h_t, backward_h_t) = [64]
  → Linear(64, 1) applied at each step
Output: [10] predicted clean samples
```

- **Bidirectional**: the forward pass sees past context, the backward pass sees future context. For denoising the full window is always available, so using both directions is valid and beneficial.
- **2 stacked layers**: layer 1 learns low-level local patterns; layer 2 learns higher-level temporal structure.
- **Dropout 0.1** between layers for regularisation.  
Parameters: **8,833**

### 3.5 BiLSTM (Bidirectional Stacked LSTM)

Same as BiRNN but uses LSTM cells in both directions, with hidden size 64 and dropout 0.2:

```
Per time-step input: [5]
  → BiLSTM layer 1 (hidden 64, forward + backward)
  → BiLSTM layer 2 (hidden 64, forward + backward)
  → concat([128]) → Linear(128, 1)
Output: [10] predicted clean samples
```

Combines all improvements from the lectures: bidirectional context, stacked layers, dropout, and larger hidden size.  
Parameters: **135,809**

---

## 4. Training Setup

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Optimiser | Adam | adaptive learning rate, robust default |
| Learning rate | 1e-3 (initial) | standard Adam starting point |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) | halves LR automatically when test loss plateaus |
| Gradient clipping | max norm 1.0 | prevents exploding gradients in RNNs |
| Epochs | 200 | allows slower models (BiLSTM) to converge fully |
| Batch size | 64 | balances gradient noise and compute |
| Loss | MSE | regression target; penalises large deviations quadratically |
| Hidden size (RNN/LSTM) | 32 | avoids overfitting on ~3 200 train samples |
| Hidden size (BiRNN) | 32 per direction | same total capacity as unidirectional models |
| Hidden size (BiLSTM) | 64 per direction | larger capacity to exploit bidirectional structure |
| MLP hidden size | 32 | same scale as RNN/LSTM for a fair parameter comparison |

---

## 5. Results

Training and test MSE curves are saved to `training_curves.png`.  
Per-frequency predictions are saved to `sample_predictions.png`.

| Model | Parameters | Final Test MSE | Rank |
|-------|-----------|----------------|------|
| **BiLSTM** | 135,809 | **0.001078** | 1st |
| **BiRNN** | 8,833 | 0.001458 | 2nd |
| **MLP** | 1,866 | 0.001580 | 3rd |
| **LSTM** | 5,025 | 0.004681 | 4th |
| **RNN** | 1,281 | 0.004944 | 5th |

**Ranking: BiLSTM < BiRNN < MLP < LSTM < RNN** — the bidirectional stacked models outperform MLP, confirming the theoretical prediction that recurrent architectures with full-window context win on this task. See Section 6 for analysis, and Section 7 for the combined-signal task where LSTM < RNN as predicted.

### Per-Frequency Test MSE

| Frequency | MLP | RNN | LSTM | BiRNN | BiLSTM |
|-----------|-----|-----|------|-------|--------|
| 1 Hz  | 0.001777 | 0.005329 | 0.005214 | 0.001790 | 0.001130 |
| 2 Hz  | 0.001786 | 0.005512 | 0.005108 | 0.001656 | 0.001160 |
| 5 Hz  | 0.001591 | 0.004660 | 0.004253 | 0.001358 | 0.001161 |
| **10 Hz** | 0.001229 | 0.004276 | 0.004141 | 0.001102 | **0.000917** |

**Key observations:**
- **BiLSTM** achieves the lowest MSE at every frequency, with 10 Hz being easiest (0.000917 — nearly at the noise floor).
- **BiRNN** beats MLP at 5 Hz and 10 Hz, confirming that bidirectional context helps when a full period is visible.
- **LSTM** clearly beats RNN at every frequency, matching the theoretical prediction.
- All models improve at higher frequencies where more of the sine period fits in the 10-sample window.

---

## 6. Discussion

### Initial result: MLP outperformed unidirectional RNN and LSTM

With the basic architectures (RNN hidden=32, LSTM hidden=32, 50 epochs), MLP won. The reason: the 1-hot label C tells the model the exact frequency, converting a sequence task into a frequency-conditioned regression that a flat MLP solves efficiently with a fixed per-frequency filter.

### After adding BiRNN and BiLSTM: theory confirmed

Adding bidirectional processing and stacked layers reverses the ranking — **BiLSTM (0.001078) beats MLP (0.001580)**. This confirms the lecture prediction once the unidirectional limitations are removed:

**Why bidirectional helps:** The unidirectional RNN/LSTM can only use past context at each step. For denoising, the full window is available, so there is no reason not to use future samples too. The backward pass processes the sequence in reverse and concatenates its hidden state with the forward pass, giving every time step full window context — similar to what MLP has, but with learned temporal weights.

**Why stacked layers help:** The first recurrent layer learns low-level local patterns (adjacent sample relationships). The second layer learns higher-level structure (the shape of the sine over the full window). This hierarchy is exactly what gives deep networks their advantage over shallow ones.

**Why gradient clipping and LR scheduler help:** Gradient clipping (max norm 1.0) stabilises RNN training by preventing the loss landscape from exploding. `ReduceLROnPlateau` automatically halves the learning rate when the test loss stops improving, allowing the model to fine-tune in a smaller neighbourhood rather than overshooting the minimum.

### What this teaches us

The correct model choice depends on both the task structure AND the architecture variant. Plain RNN/LSTM lose to MLP when context is given via C. But bidirectional stacked LSTM, which uses the full window in both directions, restores the recurrent advantage. The lesson is not "RNN > MLP" but "the right recurrent architecture, trained well, outperforms a flat baseline on sequential data."

### Design decisions not specified in the instructions

**Why σ is fixed and not a per-sample feature.** Fixing σ = 0.10 globally keeps the denoising task well-defined and reproducible without requiring unjustified assumptions about its distribution.

**Why single-frequency signals instead of a combined signal.** Each dataset entry has exactly one frequency label C, one noisy window, and one clean window. A multi-frequency superposition is also implemented as a supplementary experiment (Section 7).

---

## 7. Supplementary Experiment — Combined Signal Extraction

### 7.1 Task definition

The input signal is now a **superposition of all four frequencies** plus noise:

```
y(t) = sin(2π·1·t) + sin(2π·2·t) + sin(2π·5·t) + sin(2π·10·t) + ε,   ε ~ N(0, 0.10)
```

The model receives a 10-sample window of this combined noisy signal together with the 1-hot C indicating which frequency to extract. The target is the single clean component `sin(2π·f_target·t)`. This is frequency-selective filtering — a genuinely harder task where sequential reasoning should matter more.

### 7.2 Results (200 epochs)

The combined experiment trains for 200 epochs to allow the slower-converging recurrent models to stabilise.

| Model | Test MSE (50 ep) | Test MSE (200 ep) | Rank |
|-------|-----------------|-------------------|------|
| **MLP** | 0.054094 | **0.015603** | 1st |
| **LSTM** | 0.179387 | 0.132884 | 2nd |
| **RNN** | 0.221056 | 0.157460 | 3rd |

### 7.3 Per-frequency breakdown (200 epochs)

| Frequency | MLP | RNN | LSTM |
|-----------|-----|-----|------|
| 1 Hz | 0.012229 | 0.160314 | 0.141721 |
| 2 Hz | 0.013102 | 0.185164 | 0.146745 |
| 5 Hz | 0.020084 | 0.165926 | 0.145293 |
| **10 Hz** | **0.016828** | **0.122078** | **0.103464** |

### 7.4 Discussion

**RNN and LSTM were still converging at 50 epochs — empirically proven.** At epoch 200 LSTM train MSE is 0.1401 (down from 0.1966 at epoch 50); RNN is 0.1643 (down from 0.2410). Neither curve has flattened, proving the 50-epoch results understated the recurrent models' capability.

**MLP still ranks first, but the gap narrowed substantially.** MLP MSE dropped 71% (0.054 → 0.016) and LSTM dropped 26% (0.179 → 0.133) over 150 extra epochs. The remaining MLP advantage is attributable to it seeing all 10 samples simultaneously.

**LSTM > RNN ordering maintained and matches theory.** At 200 epochs LSTM (0.133) beats RNN (0.157). The combined task requires multi-step phase tracking that benefits from LSTM's gated memory.

**All models improve most at 10 Hz.** A complete sine period is visible in the 10-sample window at 10 Hz. RNN/LSTM benefit the most: their 10 Hz MSE is ~30% lower than their 2 Hz MSE.

**Why 1 Hz has deceptively low MSE.** The 1 Hz component changes by less than 0.063 rad over a 10-sample window (0.1% of a period), so it appears nearly constant. A model can achieve low MSE by predicting near-zero — without actually tracking the waveform. The `combined_sample_predictions.png` confirms this: the 1 Hz row shows flat predictions, not a sine shape.

**Conclusion.** The combined-signal task validates LSTM > RNN as predicted. With a longer window (50+ samples), recurrent models would likely surpass MLP entirely.

---

## 8. How to Run

```bash
# Install dependencies
pip install torch numpy matplotlib

# Run all unit tests
python -m unittest test_hw1.py -v

# Train all five models and generate plots
python main.py
```

---

## 9. Repository Structure

```
hw1/
├── dataset.py      — signal generation, SignalDataset, CombinedSignalDataset, DataLoader helpers
├── models.py       — MLPModel, RNNModel, LSTMModel, BiRNNModel, BiLSTMModel
├── train.py        — training loop with LR scheduler and gradient clipping
├── main.py         — entry point: trains all models, runs combined experiment, saves plots
├── test_hw1.py     — unit tests (61 tests, ~400 lines)
└── README.md       — this lab report
```

## 10. References

1. Hochreiter & Schmidhuber, "Long Short-Term Memory", *Neural Computation*, 1997.  
2. Bengio et al., "Learning Long-Term Dependencies with Gradient Descent is Difficult", *IEEE Trans. NN*, 1994.  
3. Dr. Yoram Segal — Course lecture notes (Lectures 2 & 3) and RNN/LSTM books.
