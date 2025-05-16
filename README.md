

# Training Nano LLaMA 3.1 Baseline

## 1. Overview

This repository provides code to train a language model on the TinyStories dataset using a custom small Transformer architecture based on LLaMA 3.1. The design can be adjusted to accommodate larger LLaMA models with 4 or 8 billion parameters. The goal is to enable training of LLaMA 3.1 efficiently on an RTX 3090 GPU.

For more details, see the LLaMA 3.1 paper: [https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)

* **Model**: 4-layer Transformer decoder with hidden size 1024, 8 attention heads, totaling 33 million parameters.
* **Tokenizer**: Vocabulary size of 10,000 tokens tailored for the TinyStories dataset.
* **Dataset**: TinyStories with about 465 million tokens for training (streaming) and 4.7 million tokens for validation.
* **Framework**: PyTorch, utilizing mixed precision training with `torch.cuda.amp`.

---

Before starting training, you can run tests with `pytest` located in `test/test_model.py` to ensure everything is working correctly.

To prepare the tokenizer files, run `prepare_dataset.py`.

## 2. Training & Evaluation

### Hyperparameters

```yaml
model_args = ModelArgs(
    dim=1024,
    n_heads=8,
    n_kv_heads=2,
    intermediate_dim=1536,
    multiple_of=256,
    ffn_dim_multiplier=1.0,
    norm_eps=1e-5,
    max_batch_size=128,
    max_seq_len=128,  # matches TinyStories sequence length
    vocab_size=10000,  # matches TinyStories vocabulary size
    n_layers=4,
    rope_theta=10000.0,
    use_scaled_rope=False,
    flash=True
)

cfg = TrainingConfig(
    batch_size=128,
    seq_len=128,
    epochs=2,
    steps_per_epoch=15000,
    report_interval=20_000_000,
    grad_clip_norm=1.0,
    learning_rate=6e-4,
    warmup_steps=10,
    max_lr=6e-4,
    min_lr=3e-5
)
```

### Token Counts

* Tokens processed per epoch: 245,760,000
* Total tokens processed over 2 epochs: 491,520,000

### Visualization

Training loss over total tokens:

![](/plot.png)

### Loss and Perplexity Summary

| Epoch | Final Training Loss | Final Training Perplexity | Final Validation Loss | Final Validation Perplexity |
| ----- | ------------------- | ------------------------- | --------------------- | --------------------------- |
| 1     | 3.3468              | 28.38                     | 2.1811                | 8.86                        |
| 2     | 2.0436              | 7.71                      | 1.9858                | 7.29                        |

<details>
<summary>Example Generated Text</summary>

* **Prompt:** "Once upon a time..."

  > "Once upon a time, there was a little girl named Lily..."

* **Prompt:** "Max had two dogs..."

  > "Max had two dogs. He was four and the other was good..."

</details>

---

## 3. How to Use

### Install Requirements

```bash
pip install -r requirement.txt
```

### Start Training

```bash
python main.py
```

---

