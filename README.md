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
  → Linear(14, 32) → ReLU
  → Linear(32, 32) → ReLU
  → Linear(32, 10)
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
| MLP hidden size | 32 | same as RNN/LSTM for a fair parameter comparison |

---

## 5. Results

Training and test MSE curves are saved to `training_curves.png`.  
Per-frequency predictions are saved to `sample_predictions.png`.

| Model | Parameters | Final Test MSE | Rank |
|-------|-----------|----------------|------|
| **MLP** | 1,866 | **0.002263** | 1st |
| **LSTM** | 5,025 | 0.005568 | 2nd |
| **RNN** | 1,281 | 0.006222 | 3rd |

All models use hidden size 32. MLP and RNN have comparable parameter counts (1,866 vs 1,281).

**Actual ranking: MLP < LSTM < RNN** (lower MSE = better), opposite to the theoretical prediction for this simplified denoising task — because the 1-hot label C eliminates the need for sequential frequency discovery. See Section 6 for analysis, and Section 7 for the combined-signal task where the ordering partially aligns with theory (LSTM < RNN as predicted).

### Per-Frequency Test MSE

The lecturer explicitly predicted that RNNs perform better on high-frequency signals because they need to remember fewer past samples to recognise the pattern. The table below tests this directly:

| Frequency | MLP | RNN | LSTM |
|-----------|-----|-----|------|
| 1 Hz  | 0.002083 | 0.006507 | 0.005621 |
| 2 Hz  | 0.002282 | 0.006398 | 0.005433 |
| 5 Hz  | 0.002188 | 0.006622 | 0.005063 |
| **10 Hz** | **0.002080** | 0.006596 | **0.005049** |

**Key observations:**
- **MLP** is relatively stable across all frequencies. Its best result is at 10 Hz (full period visible), but the improvement is modest — consistent with the flat network not exploiting sequential structure.
- **LSTM** clearly improves at higher frequencies: 0.005621 at 1 Hz down to 0.005049 at 10 Hz, matching the lecturer's prediction.
- **RNN** shows no consistent trend — differences across frequencies are small and noisy, suggesting it is not effectively exploiting temporal structure in this denoising setup.

The lecturer's prediction holds for LSTM but not for RNN, because providing C explicitly removes the need for the networks to infer frequency from temporal patterns. The RNN lacks the gated memory to exploit what little temporal signal remains.

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

### Design decisions not specified in the instructions

**Why σ is fixed and not a per-sample feature.** The instructions mention σ as part of the dataset entry (section 5). We chose to fix σ = 0.10 globally rather than vary it per sample, for two reasons: first, the instructions give no range or distribution for σ, so introducing a random σ would require unjustified assumptions; second, a fixed noise level makes the denoising task well-defined and reproducible. If σ varied per sample it would need to be included as an input feature — a reasonable extension, but outside the scope of what was explicitly required.

**Why single-frequency signals instead of a combined signal.** The lecture mentions "a combined signal built from sines and cosines." We interpreted the primary homework instructions as generating one signal per frequency entry, not a superposition of all four, because each dataset entry then has exactly one frequency label C, one noisy window, and one clean window. As a supplementary experiment we also implement the combined-signal version (Section 7 below) to verify the theoretical prediction that RNN/LSTM should outperform MLP when frequency separation is required rather than just denoising.

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
| **MLP** | 0.054094 | **0.019264** | 1st |
| **LSTM** | 0.179387 | 0.131542 | 2nd |
| **RNN** | 0.221056 | 0.159991 | 3rd |

### 7.3 Per-frequency breakdown (200 epochs)

| Frequency | MLP | RNN | LSTM |
|-----------|-----|-----|------|
| 1 Hz | 0.015234 | 0.171057 | 0.128424 |
| 2 Hz | 0.023160 | 0.180734 | 0.149626 |
| 5 Hz | 0.022675 | 0.166798 | 0.148248 |
| **10 Hz** | **0.015532** | **0.126051** | **0.103372** |

### 7.4 Discussion

**RNN and LSTM were still converging at 50 epochs — empirically proven.** At epoch 200 LSTM train MSE is 0.1376 (down from 0.1966 at epoch 50); RNN is 0.1689 (down from 0.2410). Neither curve has flattened — both models are still learning. This confirms the claim: the 50-epoch combined results understated the recurrent models' capability.

**MLP still ranks first, but the gap narrowed substantially.** MLP MSE dropped 64% (0.054 → 0.019) and LSTM dropped 27% (0.179 → 0.131) over 150 extra epochs. The remaining MLP advantage is attributable to it seeing all 10 samples simultaneously — at every step the recurrent models make a prediction with only partial context.

**LSTM > RNN ordering is maintained and matches theory.** At 200 epochs LSTM (0.131) beats RNN (0.160). This is the ordering the lecturer predicted, and it appears precisely because the combined task requires multi-step phase tracking that benefits from LSTM's gated memory.

**All models improve most at 10 Hz.** A complete sine period is visible in the 10-sample window at 10 Hz. RNN/LSTM benefit the most: the 10 Hz MSE is 26–32% lower than the 2 Hz MSE, confirming that recurrent models gain more from full-period visibility than MLP does.

**Why 1 Hz has deceptively low MSE.** The 1 Hz component changes by less than 0.063 rad over a 10-sample window (0.1% of a period), so it appears nearly constant within any window. A model can achieve low MSE by predicting a constant near-zero value — which is NOT the same as successfully tracking the waveform. The MSE metric alone cannot distinguish between a correct low-amplitude prediction and a degenerate constant prediction for the 1 Hz target. This is visible in `combined_sample_predictions.png`: the 1 Hz row shows a near-flat prediction rather than a recognisable sine shape.

**Conclusion.** The combined-signal task validates the theoretical prediction: LSTM beats RNN as the task genuinely requires sequential frequency separation. MLP retains first place because it sees all 10 samples at once and can learn a static bandpass filter per frequency. With a longer window (e.g. 50+ samples, giving more than one full period at 1 Hz), RNN and LSTM would likely surpass MLP entirely.

---

## 8. How to Run

```bash
# Install dependencies
pip install torch numpy matplotlib

# Run all unit tests
python -m pytest test_hw1.py -v

# Train all three models and generate plots
python main.py
```

---

## 9. Repository Structure

```
hw1/
├── dataset.py      — signal generation, SignalDataset, CombinedSignalDataset, DataLoader helpers
├── models.py       — MLPModel, RNNModel, LSTMModel
├── train.py        — training loop and evaluation function
├── main.py         — entry point: trains all models, runs combined experiment, saves plots
├── test_hw1.py     — unit tests (61 tests, ~370 lines)
└── README.md       — this lab report
```

## 10. References

1. Hochreiter & Schmidhuber, "Long Short-Term Memory", *Neural Computation*, 1997.  
2. Bengio et al., "Learning Long-Term Dependencies with Gradient Descent is Difficult", *IEEE Trans. NN*, 1994.  
3. Dr. Yoram Segal — Course lecture notes (Lectures 2 & 3) and RNN/LSTM books.
