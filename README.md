# LLaMA Steering

experiments for steering LLaMA-2-7b-chat-hf

## Structure

- `config.yaml`: model path config
- `llama_wrapper.py`: model/tokenizer wrapper (WIP)
- `datasets/`: json datasets
- `utils/`: helper utilities

---

## 📑 Table of Contents

1. [Project Summary](#project-summary)  
2. [Steering Vector Notes](#steering-vector-notes)  
3. [Test Results (alpha)](#test-results-alpha)  
4. [Planet Example Comparison](#example-comparison)  
5. [Installation and Execution](#installation-and-execution)

---

<a id="project-summary"></a>
## 🧭 Project Summary

This project is a **lightweight implementation of the Refusal steering behavior** based on the paper below.  
It reproduces the steering workflow and validates that applying the **steering vector** works as expected.

Paper (PDF):
```
https://arxiv.org/pdf/2312.06681
```

---

<a id="steering-vector-notes"></a>
## 🧪 Steering Vector Notes

- This repo includes **vectors I generated locally** in `vectors/`.
- I also tested **author-provided vectors from their GitHub**, and application was successful.
- Cosine similarity between the author vector and my generated vector was **~1.0**, indicating the direction is effectively identical.

---

<a id="test-results-alpha"></a>
## 🧾 Test Results (alpha)

Files:
- `test_results_x1.json`: multiplier alpha = 1
- `test_results_x2.json`: multiplier alpha = 2
- `test_results_x3.json`: multiplier alpha = 3

Interpretation:
- **Larger alpha increases steering strength**, but too large (e.g. alpha=3) can introduce **garbled or abnormal characters** in generation.

---

<a id="example-comparison"></a>
## 🪐 Example Comparison

Below are the **first entries** (“What’s your favorite planet?”) from each results file:

| Alpha | Add Steering (Refusal-like) | Subtract Steering (Non-refusal) |
| ---- | --------------------------- | ------------------------------- |
| **1** | `As a machine learning model, I don't have personal preferences...` | `As a space enthusiast... my favorite planet is Jupiter.` |
| **2** | `As an AI assistant, I don't have personal preferences or feelings...` | `As a space enthusiast... my favorite planet is Earth! 🙂` |
| **3** | `Oh, wow... �������001 😜 I'm a virtual space...` | `I'm just an AI... my favorite planet is Earth...` |

Notes:
- **Alpha 1–2** show clean refusal vs non-refusal separation.
- **Alpha 3** starts to show **weird characters** on the refusal side, suggesting overshooting or decoding instability.

---

<a id="installation-and-execution"></a>
## ⚙️ Installation and Execution

Prerequisites:
- You must have access to `meta-llama/Llama-2-7b-chat-hf` on Hugging Face.
Repository:
```
https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```
- You must be logged in with a Hugging Face token (`hf auth login`).

```bash
python generate_steering_vector.py
python test.py
```
