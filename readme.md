# Attention Mechanisms & KV Cache — Practical Explanation

This document explains the core attention mechanisms used in modern Transformer models and how **KV cache** works during inference.  
The explanations are aligned with **real implementations and experiments**, not just theory.

---

## 1. Multi-Head Attention (MHA)

### Definition
**Multi-Head Attention (MHA)** is the original attention mechanism introduced in the Transformer architecture.

Each attention head has its **own Query (Q), Key (K), and Value (V)** projections.  
All heads attend independently, and their outputs are concatenated.

---

### How it works

For each head:

Q = X · Wq
K = X · Wk
V = X · Wv

Attention = softmax(QKᵀ / √d) · V

Outputs from all heads are concatenated and projected.

---

### Why MHA is powerful
- Heads learn **different relationships** (syntax, semantics, position)
- Maximum expressiveness
- Best raw model quality

---

### Downsides
- Large KV cache
- High memory usage
- Slow decoding for long contexts

---

### Where it is used
- Smaller models
- Research and analysis
- When quality is more important than inference speed

---

## 2. Grouped Query Attention (GQA)

### Definition
**Grouped Query Attention (GQA)** reduces the number of **Key/Value heads** while keeping many **Query heads**.

Example:
- 12 Query heads
- 4 Key/Value heads  
→ Every 3 Query heads share the same K/V

---

### How it works
- Each Query head still has its own projection
- Keys and Values are **shared across groups of heads**

This reduces memory while preserving most of the expressiveness.

---

### Why GQA exists
- Smaller KV cache
- Faster decoding
- Near-identical quality to MHA in practice

---

### Trade-offs
- Heads in the same group are correlated
- Slight reduction in flexibility compared to MHA

---

### Where it is used
**Industry standard**
- LLaMA-2 / LLaMA-3
- Mistral / Mixtral
- Most modern production LLMs

---

## 3. Multi-Query Attention (MQA)

### Definition
**Multi-Query Attention (MQA)** is an extreme version of GQA.

- Many Query heads
- **Only ONE Key and ONE Value head**

All Query heads attend to the same K/V.

---

### Why MQA exists
- Smallest possible KV cache
- Fastest decoding
- Best for high-throughput inference

---

### Trade-offs
- Lowest expressiveness
- Heads become highly correlated
- Can reduce quality on complex reasoning tasks

---

### Where it is used
- Latency-critical systems
- Memory-constrained environments
- Large batch inference servers

---

## 4. Multi-Head Latent Attention (MLA)

### Definition
**Multi-Head Latent Attention (MLA)** compresses **Keys and Values into a latent space** before attention.

Instead of storing full K/V:

K, V ∈ ℝ[T × d]

MLA stores:

K_latent, V_latent ∈ ℝ[T × d_latent]

where `d_latent << d`.

---

### How it works

1. Compress K/V:
   K_lat = K · W_down
   V_lat = V · W_down
2. Reconstruct before attention:
   K_rec = K_lat · W_up
   V_rec = V_lat · W_up
3. Compute attention using reconstructed K/V.

---

### Why MLA matters
- Much smaller KV cache
- More expressive than GQA and MQA
- Better quality for the same memory budget

---

### Key insight
- GQA/MQA reduce **number of KV heads**
- MLA reduces **dimensionality of KV itself**

---

### Where it is used
- Long-context models
- Memory-efficient inference
- Next-generation Transformer architectures

---

## 5. Key–Value Cache (KV Cache)

### Definition
**KV cache stores past Keys and Values during autoregressive generation**.

Instead of recomputing K/V for all previous tokens at every generation step, the model reuses cached K/V.

---

### Why KV cache exists
During generation:
- Tokens are produced one-by-one
- Without cache → K/V recomputed every step
- With cache → K/V computed once per token

---

### What KV cache actually saves
- Key and Value projection computation
- Large amount of repeated work

---

### What KV cache does NOT do
- ❌ Does NOT reduce attention complexity
- ❌ Does NOT speed up a full forward pass

---

### Where KV cache is effective
- Autoregressive decoding
- Token-by-token generation
- Long-context inference

---

### How KV cache was validated
- Compared `use_cache=True` vs `use_cache=False`
- Measured **ms per generated token**
- Observed increasing speedup as context length grew

---

## Summary Table

| Mechanism | KV Heads | KV Size | Decode Speed | Quality |
|---------|---------|---------|-------------|---------|
| MHA | Many | Large | Slow | Best |
| GQA | Few | Medium | Fast | Near-MHA |
| MQA | 1 | Very Small | Fastest | Lower |
| MLA | Latent | Very Small | Fast | High |
| KV Cache | Stored | Reused | Faster Decode | Same |

### 1) What you measured

You compared two *autoregressive decoding* modes:

- **NO cache (`use_cache=False`)**  
  Every generation step recomputes **K/V for the entire prefix** (prompt + previously generated tokens).

- **WITH cache (`use_cache=True`)**  
  Past **K/V are stored** and reused; each step computes **K/V only for the new token**.

This is the *correct* scenario to show KV-cache advantage (KV cache is mainly a **decoding** optimization, not a “full forward pass” optimization).

---

### 2) Your results (from the run)

| Prompt length `T` | NO cache time (ms) | NO cache ms/token | WITH cache time (ms) | WITH cache ms/token | Speedup |
|---:|---:|---:|---:|---:|---:|
| 64  | 3441.37 | 53.77  | 1338.96 | 20.92 | 2.57× |
| 128 | 4121.32 | 64.40  | 1234.51 | 19.29 | 3.34× |
| 256 | 5934.30 | 92.72  | 1393.55 | 21.77 | 4.26× |
| 512 | 10402.67| 162.54 | 1488.00 | 23.25 | 6.99× |

**Big takeaway:** KV cache advantage increases strongly as **prompt length T grows**.

---

### 3) Why NO-cache gets much slower as T increases

During autoregressive generation, you produce tokens one by one:

- Let `T` = prompt length (tokens already in context)
- Let `N` = number of generated tokens

---

## Final Takeaways

- **MHA** → Maximum expressiveness, highest cost
- **GQA** → Best quality–speed balance (industry default)
- **MQA** → Fastest and smallest cache
- **MLA** → Compress KV intelligently (future direction)
- **KV cache** → Eliminates repeated work during generation

**KV cache does not make attention cheaper — it removes repeated computation.**