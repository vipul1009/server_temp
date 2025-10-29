from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Optional: safer CUDA error reporting
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ✅ Use a valid model ID (Gemma-3 27B not yet public)
model_id = "google/gemma-3-27b-it"  # Replace with 27B when available

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model in 4-bit quantized mode
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",              # Automatically maps to GPU
    load_in_4bit=True,              # 4-bit quantization
    torch_dtype=torch.float16       # Efficient precision
)

# Prompt
prompt = "Explain the concept of quantization in NLP."

# Tokenize and move to GPU
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate output with safe sampling
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=1.0,     # Safer range
        top_p=0.9,
        top_k=50
    )

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
