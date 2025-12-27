import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset

# 1. Environment & Data Setup
model_name = 'google/flan-t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
dataset = load_dataset("knkarthick/dialogsum")

# Select a sample for testing
target_index = 200
dialogue = dataset['test'][target_index]['dialogue']
ground_truth = dataset['test'][target_index]['summary']

# 2. Prompt Strategies Implementation (Zero, One, Few)
# Zero-Shot
zero_shot_prompt = f"Summarize the following conversation.\n\n{dialogue}\n\nSummary:"

# One-Shot (using example at index 40)
one_shot_prompt = (
    f"Dialogue:\n{dataset['test'][40]['dialogue']}\nSummary:\n{dataset['test'][40]['summary']}\n\n"
    f"Dialogue:\n{dialogue}\nSummary:"
)

# Few-Shot (using examples at 40, 80, 120)
few_shot_prompt = ""
for idx in [40, 80, 120]:
    few_shot_prompt += f"Dialogue:\n{dataset['test'][idx]['dialogue']}\nSummary:\n{dataset['test'][idx]['summary']}\n\n"
few_shot_prompt += f"Dialogue:\n{dialogue}\nSummary:"

# 3. Generation Function with Config Support
def generate_summary(prompt, config):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs["input_ids"], generation_config=config)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 4. Parameter Impact Testing (Configurations)
# Test A: Default (Greedy)
greedy_config = GenerationConfig(max_new_tokens=50)

# Test B: High Creativity (High Temperature)
creative_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.9)

# Test C: High Precision (Low Temperature - Best for Legal/Formal)
precise_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.1)

# 5. Execution and Final Comparison
print("--- PROMPTING STRATEGIES ---")
print(f"Zero-Shot Result: {generate_summary(zero_shot_prompt, precise_config)}")
print(f"One-Shot Result:  {generate_summary(one_shot_prompt, precise_config)}")
print(f"Few-Shot Result:  {generate_summary(few_shot_prompt, precise_config)}")

print("\n--- PARAMETER IMPACT ON GENERATION ---")
print(f"Greedy Decoding:    {generate_summary(few_shot_prompt, greedy_config)}")
print(f"Creative (T=0.9):   {generate_summary(few_shot_prompt, creative_config)}")
print(f"Precise (T=0.1):    {generate_summary(few_shot_prompt, precise_config)}")

print(f"\nREFERENCE SUMMARY: {ground_truth}")

#exercise1
prompt = f"""
Can you tell me what happened in this chat in one sentence?

{dialogue}

Summary:
"""

prompt = f"""
Summarize the following conversation.

{dialogue}
"""
prompt = f"""
{dialogue}

TL;DR:
"""
#exercise2
generation_config = GenerationConfig(
    max_new_tokens=50, 
    do_sample=True, 
    temperature=0.1
)
inputs = tokenizer(few_shot_prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        generation_config=generation_config,
    )[0], 
    skip_special_tokens=True
)

print(dash_line)
print(f'MODEL GENERATION - FEW SHOT:\n{output}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
