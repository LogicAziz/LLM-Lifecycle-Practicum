# 🚀 Generative AI with Large Language Models (LLMs) - Lab Solutions

This repository contains my practical implementations and technical notes for the **Generative AI with LLMs** course labs. The project covers the entire LLM lifecycle, from prompt engineering to fine-tuning and model evaluation.

---

## 📁 Lab 1: Dialogue Summarization & Prompt Engineering

### 📋 Overview
In this lab, I explored the use of **FLAN-T5** from Hugging Face for dialogue summarization. The goal was to understand how different prompting strategies affect the model's output quality and how to control the generation process.

### 🚀 Key Techniques Implemented

* **Tokenization:** Converting raw text into input IDs using the `AutoTokenizer`.
* **Inference Strategies:**
    * **Zero-Shot:** Testing the model's baseline ability to summarize without examples.
    * **One-Shot:** Providing a single (Dialogue -> Summary) pair to guide the model.
    * **Few-Shot:** Providing multiple examples to reinforce the desired pattern and style.
* **Configuration Tuning:** * Adjusting `temperature` to control creativity vs. precision.
    * Using `do_sample`, `top_k`, and `top_p` for advanced decoding.

### 🛠️ Core Functions
One of the main components is the `make_prompt` function, which dynamically builds a structured prompt for Few-Shot learning:

```python
def make_prompt(example_indices_full, example_index_to_summarize):
    # Iterates through examples and builds a structured prompt 
    # concluding with the target dialogue for the model to summarize.
