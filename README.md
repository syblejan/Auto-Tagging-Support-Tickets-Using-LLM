🎫 Auto-Tagging Support Tickets Using LLM
Automatically classify customer support tickets into predefined categories using a local open-source LLM — comparing Zero-Shot, Few-Shot, and QLoRA Fine-Tuned approaches.

📌 Objective
Customer support teams receive thousands of tickets daily, and manually routing each one to the correct team is slow and error-prone. This project automates the ticket categorization process by leveraging Qwen2.5-3B-Instruct, an open-source large language model, to predict the most relevant support category (tag) for any incoming ticket — without relying on any paid API.
The model is asked to output a ranked Top-3 list of predicted tags in structured JSON, enabling flexible routing even when the correct category is ambiguous.

🧪 Methodology / Approach
Dataset

Source: Kaggle — Customer Support Ticket Dataset
Columns used: Ticket Description (input), Ticket Type (ground-truth label)
Split: 400 training samples, 50 holdout test samples (sampled from 450 total)

Model
PropertyValueModelQwen/Qwen2.5-3B-InstructQuantization (inference)4-bit NF4 via BitsAndBytesPrecision (fine-tuning)Float16PlatformGoogle Colab (GPU)
Prompting Strategy
All three approaches use a structured prompt that instructs the model to return only a valid JSON object of the form:
json{ "top_3_tags": ["Tag1", "Tag2", "Tag3"] }
Temperature is kept very low (0.1) to enforce deterministic, structured output.

Approach 1 — Zero-Shot Classification
The base model is given only the ticket text and the full list of available tags, with no examples. It must classify the ticket purely from its pre-trained understanding of language.
Approach 2 — Few-Shot Classification
One representative example ticket is sampled per category from the training set and appended to the prompt as context. The model uses these demonstrations to better align its output with the expected label format.
Approach 3 — QLoRA Fine-Tuning
The model is fine-tuned on 400 labeled tickets using LoRA adapters attached to all major projection layers (q_proj, k_proj, v_proj, o_proj, and MLP layers).
HyperparameterValueLoRA rank (r)16LoRA alpha32LoRA dropout0.05Learning rate2e-4Batch size2 (effective: 8 with gradient accumulation)Training steps40OptimizerAdamW (PyTorch native)Max sequence length512 tokens
Training data is formatted as complete chat conversations (system → user → assistant) using the model's built-in chat template, ensuring the fine-tuned adapter learns the exact output format required.

📊 Key Results & Observations
Evaluation is performed on the same 50 holdout test tickets across all three approaches, measuring:

Top-1 Accuracy — correct label is the model's first prediction
Top-3 Accuracy — correct label appears anywhere in the model's top-3 predictions
JSON Parse Failures — responses that could not be parsed (indicates prompt adherence)

ApproachTop-1 AccuracyTop-3 AccuracyZero-ShotBaselineBaselineFew-Shot+ vs Zero-Shot+ vs Zero-ShotFine-Tuned (QLoRA)HighestHighest
Key Observations

Zero-Shot provides a reasonable baseline, demonstrating that Qwen2.5-3B has strong enough language understanding to infer ticket intent without examples.
Few-Shot improves over Zero-Shot by anchoring the model to the exact label vocabulary and format, reducing ambiguous or off-label predictions.
QLoRA Fine-Tuning yields the best results with near-zero JSON parse failures, showing that even 40 steps of supervised fine-tuning on 400 samples meaningfully specializes the model for this structured output task.
Structured JSON output (rather than free-text classification) proved critical — it makes post-processing reliable and enables multi-label routing without additional parsing logic.
The use of 4-bit quantization for inference and float16 LoRA for training made the entire pipeline runnable on a free Colab T4 GPU.
