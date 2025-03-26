import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
# the cool_fusion function are in source/cool_fusion.py
from source.cool_fusion import CoolFusion
import pandas as pd
from tqdm import tqdm
import os

from huggingface_hub import login
login("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")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch.cuda.empty_cache()

device = torch.device("cuda")
# Define the model names
model_name1 = "meta-llama/Llama-3.2-1B-Instruct"
model_name2 = "Qwen/Qwen2.5-1.5B-Instruct"
# Load tokenizers
tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
tokenizer2 = AutoTokenizer.from_pretrained(model_name2)

# Load models (using CPU by default; add device_map="auto" for GPU)
model1 = AutoModelForCausalLM.from_pretrained(model_name1).to(device)
model2 = AutoModelForCausalLM.from_pretrained(model_name2,trust_remote_code=True).to(device)

# Sample prompt for generation



# usage

fused_model = CoolFusion(
    models={"1": model1, "2": model2},
    tokenizers={"1": tokenizer1, "2": tokenizer2}
)


def format_prompt(row):
    prompt = f"Question: {row[0]}\nChoices:\n"
    for i,option in enumerate(['A', 'B', 'C', 'D']):
        prompt += f"{option}. {row[i+1]}\n"
    prompt += "Answer:"
    return prompt
def format_answer(row):
    return row[-2]
def extract_answer(output_text):
    # Very basic extraction — adjust for your model's output style
    for choice in ['A', 'B', 'C', 'D']:
        if choice in output_text:
            return choice  
    return None


def evaluate_subjects(directory, fused_model):
    
    subject_paths = [(os.path.join(directory,f),f[:-4]) for f in os.listdir(directory) if f.endswith(".csv")  ]
    results = {}
    for subject_path,subject_name in subject_paths:
        results[subject_name] = evaluate_subject(subject_path,fused_model)
        print(results[subject_name])
    return results
def evaluate_subject(subject_path,fused_model):
    df = pd.read_csv(subject_path)
    df["prompt"] = df.apply(format_prompt, axis=1)
    df["answer"] = df.apply(format_answer, axis=1)
    correct = 0
    total = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(subject_path)):
        input = row['prompt']
        
        output = fused_model.generate(input,max_length=len(input.split()) + 5,verbose=False)
        
        pred = extract_answer(output[len(row['prompt']):])  # trim prompt from generated
        
        if pred == row['answer']:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy


def evaluate_individual_bench(directory, model, tokenizer):
    subject_paths = [(os.path.join(directory,f),f[:-4]) for f in os.listdir(directory) if f.endswith(".csv")  ]
    results = {}
    for subject_path,subject_name in subject_paths:
        results[subject_name] = evaluate_individual(subject_path,model, tokenizer)
        print(results[subject_name])
    return results
def evaluate_individual(subject_path, model, tokenizer):
    df = pd.read_csv(subject_path)
    df["prompt"] = df.apply(format_prompt, axis=1)
    df["answer"] = df.apply(format_answer, axis=1)
    correct = 0
    total = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(subject_path)):
        input = row['prompt']
        pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
        sequences = pipeline(max_new_tokens=10, do_sample=False,eos_token_id=tokenizer.eos_token_id, text_inputs=input)
        
        generated_part = sequences[0]["generated_text"]
        
        pred = extract_answer(generated_part)  # trim prompt from generated
        
        if pred == row['answer']:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy
