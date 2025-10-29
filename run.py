from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Optional: safer CUDA error reporting
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ✅ Use a valid public model (Gemma 27B not released yet)
model_id = "google/gemma-27b-it"

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

# ✅ Greedy decoding to avoid sampling errors
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False  # disables sampling, avoids CUDA multinomial crash
    )

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# Optional: Save output to file
with open("gemma_output.txt", "w") as f:
    f.write(response)
