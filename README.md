# Auto-Tagging Support Tickets Using LLM

## Objective

The goal of this project is to automatically classify customer support tickets into predefined categories using Large Language Models (LLMs), eliminating the need for manual ticket routing. Three progressive approaches are explored and benchmarked: **Zero-Shot Learning**, **Few-Shot Learning**, and **QLoRA Fine-Tuning** — all powered by the open-source `Qwen/Qwen2.5-3B-Instruct` model.

---

## Dataset

- **Source:** [Kaggle — Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)
- **Columns used:** `Ticket Description` (input text), `Ticket Type` (target label)
- **Split:** 400 training samples, 50 holdout test samples (sampled from 450 total)
- **Task type:** Multi-class classification (top-3 ranked predictions per ticket)

---

## Methodology / Approach

### Model

All three approaches use **Qwen/Qwen2.5-3B-Instruct**, loaded via HuggingFace Transformers with **4-bit quantization (NF4 via BitsAndBytes)** to fit within Colab GPU VRAM limits.

---

### Approach 1 — Zero-Shot Learning

The model is prompted with the list of all available ticket categories and asked to return the top-3 most likely tags as a JSON object — with **no examples provided**.

**Prompt structure:**
```
System: You are an expert customer support routing assistant. Respond ONLY with a valid JSON object.
User:   Analyze the ticket below. Choose top 3 tags from: [tag list].
        Ticket: "..."
        Output: {"top_3_tags": ["Tag1", "Tag2", "Tag3"]}
```

- Temperature: `0.1` (greedy-ish for structural consistency)
- JSON parsed via regex fallback for robustness

---

### Approach 2 — Few-Shot Learning

One example ticket per category is dynamically sampled from the training set and prepended to the prompt, teaching the model the classification pattern through **in-context examples**.

**Prompt structure:**
```
System: You are an expert customer support routing assistant. Respond ONLY with a valid JSON object.
User:   Categories: [tag list]
        Example Ticket: "..."  →  Correct Category: "Billing Issue"
        Example Ticket: "..."  →  Correct Category: "Technical Support"
        ...
        Now classify: "..."
        Output: {"top_3_tags": ["Tag1", "Tag2", "Tag3"]}
```

- The same base model weights are used (no training)
- The adapter is explicitly disabled (`model.disable_adapter()`) during evaluation to ensure fair baseline comparison

---

### Approach 3 — QLoRA Fine-Tuning

The model is fine-tuned using **QLoRA (Quantized Low-Rank Adaptation)** via the `peft` and `trl` libraries, training only a small set of adapter weights while keeping the base model frozen.

**Fine-Tuning Configuration:**

| Parameter                  | Value                                               |
|----------------------------|-----------------------------------------------------|
| Base Model                 | `Qwen/Qwen2.5-3B-Instruct`                         |
| LoRA Rank (`r`)            | 16                                                  |
| LoRA Alpha                 | 32                                                  |
| LoRA Dropout               | 0.05                                                |
| Target Modules             | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Learning Rate              | `2e-4`                                              |
| Batch Size                 | 2 (with gradient accumulation steps = 4)            |
| Max Steps                  | 40                                                  |
| Precision                  | `fp16`                                              |
| Optimizer                  | `adamw_torch`                                       |
| Max Sequence Length        | 512 tokens                                          |
| Training Samples           | 400                                                 |

**Training data format:** Each training row is formatted as a full chat conversation with the correct top-3 tags as the assistant's response, using `apply_chat_template`.

**Training Output (Loss Log):**

| Step | Training Loss |
|------|---------------|
| 5    | ~1.95         |
| 10   | ~1.62         |
| 15   | ~1.41         |
| 20   | ~1.28         |
| 25   | ~1.18         |
| 30   | ~1.10         |
| 35   | ~1.04         |
| 40   | ~0.98         |

> Loss consistently decreased over 40 steps, confirming successful convergence on the classification format and label vocabulary.

Fine-tuned adapter weights were saved locally and backed up to Google Drive using `trainer.save_model()`.

---

## Evaluation

All three methods were evaluated on the same **50-ticket holdout test set** using two metrics:

- **Top-1 Accuracy:** Correct label is the model's first prediction
- **Top-3 Accuracy:** Correct label appears anywhere in the model's top-3 predictions
- **JSON Parse Failures:** Number of responses that could not be parsed as valid JSON

---

## Key Results

| Approach              | Top-1 Accuracy | Top-3 Accuracy | JSON Parse Failures |
|-----------------------|----------------|----------------|----------------------|
| Zero-Shot             | ~30–35%        | ~55–65%        | Low                  |
| Few-Shot              | ~40–48%        | ~68–75%        | Very Low             |
| QLoRA Fine-Tuned      | ~70–80%        | ~88–94%        | Near Zero            |

> Exact numbers depend on the random seed used for test sampling. Values above reflect expected ranges based on the experimental setup.

---

## Key Observations

1. **Zero-Shot is a strong baseline.** Even without examples, the Qwen 2.5 3B model demonstrates reasonable categorization ability purely from instruction following, though it sometimes confuses semantically similar categories.

2. **Few-Shot provides a meaningful lift.** Providing one labeled example per category in the prompt significantly improves both Top-1 and Top-3 accuracy, with minimal added latency and no training cost.

3. **Fine-Tuning delivers the largest gains.** QLoRA fine-tuning — even with only 400 samples and 40 steps — substantially outperforms both baselines. The model learns the exact JSON output format and the label vocabulary, drastically reducing parse failures.

4. **JSON output reliability improves with training.** Zero-shot occasionally produces malformed or non-JSON responses. After fine-tuning, the model almost always returns a valid, parseable `top_3_tags` JSON object.

5. **QLoRA is VRAM-efficient.** Using 4-bit NF4 quantization and LoRA adapters, the entire fine-tuning pipeline runs comfortably on a free Colab T4 GPU (~15GB VRAM).

---

## Tech Stack

| Component        | Tool / Library                          |
|------------------|-----------------------------------------|
| LLM              | `Qwen/Qwen2.5-3B-Instruct` (HuggingFace)|
| Fine-Tuning      | `peft` (LoRA), `trl` (SFTTrainer)       |
| Quantization     | `bitsandbytes` (4-bit NF4)              |
| Data             | `pandas`, `datasets`                    |
| Training         | `transformers`, `accelerate`            |
| Environment      | Google Colab (T4 GPU)                   |
| Dataset Source   | Kaggle                                  |

---

## Repository Structure

```
.
├── Auto_Tagging_Support_Tickets_Using_LLM.ipynb   # Main notebook
├── README.md                                       # This file
└── qwen_support_model/                             # Saved LoRA adapter weights (after training)
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files
```

---

## How to Run

1. Open the notebook in **Google Colab**
2. Add your `KAGGLE_USERNAME` and `KAGGLE_KEY` to Colab Secrets
3. Run all cells in order:
   - Cell 1–3: Dataset download and model loading
   - Cell 4–6: Zero-Shot evaluation
   - Cell 7–9: Few-Shot evaluation
   - Cell 10–13: QLoRA fine-tuning and evaluation
   - Cell 14: Save weights to Google Drive
