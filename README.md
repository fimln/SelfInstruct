# Efficient Self-Instruct: Aligning Language Models with Self-Generated Instructions Using Smaller Models

**Final Project Submission for Kapita Selekta Ilmu Komputer (KSNLP), Magister Ilmu Komputer, Universitas Gadjah Mada**

**Group Onde Mande**:
1. Kevin Pratama - 25/574311/PPA/07241
2. Alfi Maulana Akbar - 25/574622/PPA/07253
3. Maulana Yusuf Habibi - 25/575185/PPA/07275

Replicating Self-Instruct ([Wang et al., 2023](https://aclanthology.org/2023.acl-long.754/)) with a small **4B** open model (Qwen3.5-4B via Ollama) instead of GPT-3 davinci (175B), as both the data generator and the fine-tuning target. The goal: does the Self-Instruct effect survive when the generator is roughly **44x smaller** than in the original paper?

## What it does

The pipeline runs the four canonical Self-Instruct steps, then fine-tunes and evaluates:

1. **Seed selection**, 175 seeds chosen automatically via K-Means clustering over Dolly-15k embeddings (instead of hand-written seeds).
2. **Instruction generation**, few-shot prompt the model to author new instructions.
3. **Classification identification**, label each task as classification or open-ended.
4. **Instance generation**, produce one input/output pair per instruction (output-first for classification).
5. **Filtering**, heuristics + ROUGE-L overlap (< 0.7) + embedding similarity (< 0.85) deduplication.
6. **Fine-tuning**, LoRA via Unsloth on Qwen3.5-4B (single RTX 4090), then ROUGE-L evaluation.

## Results

Fine-tuning on 970 self-generated instances:

| Evaluation | Base | Fine-tuned | Delta |
|---|---|---|---|
| **SuperNI subset** (zero-shot, paper metric) | 33.8 | **45.4** | **+11.6** |
| In-distribution held-out | 0.1863 | 0.5370 | +0.3507 |

The +11.6 gain on SuperNI (tasks the generator never produced) is the key evidence that Self-Instruct works at 4B scale.

## Repo contents

| Path | Description |
|---|---|
| `selfinstruct_lora_vast_v3.ipynb` | Full pipeline: generation, filtering, LoRA fine-tuning, evaluation |
| `generated_data.jsonl` | Raw generated instructions + instances (1,308 instructions) |
| `finetuning_data.jsonl` | Final 970 `{prompt, completion}` pairs used for fine-tuning |
| `manual_review_sample.jsonl` | Instruction samples for manual review |

> LoRA checkpoints (`outputs/`) are not tracked in git due to size.

## How to run

The pipeline lives entirely in the notebook, run top to bottom. Tahap 1 (data generation) needs Ollama, fine-tuning runs on a GPU (the notebook targets vast.ai / RTX 4090).

```bash
# Generator (Tahap 1)
ollama pull qwen3.5:4b
ollama serve

pip install ollama sentence-transformers scikit-learn rouge-score datasets numpy
jupyter notebook selfinstruct_lora_vast_v3.ipynb
```

Fine-tuning dependencies (Unsloth, torch, trl) are installed from within the notebook.
