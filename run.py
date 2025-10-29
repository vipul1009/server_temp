from transformers import AutoProcessor, AutoModelForImageTextToText

# Load model and processor
processor = AutoProcessor.from_pretrained("google/gemma-3-27b-it")
model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-27b-it")

# Input question
text = "What is SVM in machine learning?"

# Tokenize input
inputs = processor(text, return_tensors="pt").to(model.device)

# Generate with additional parameters
outputs = model.generate(
    **inputs,
    max_new_tokens=150,       # Max tokens to generate
    temperature=0.7,          # Creativity level (lower = more focused)
    top_p=0.9,                # Nucleus sampling
    top_k=50,                 # Limits sampling to top-k probable tokens
    repetition_penalty=1.1,   # Avoid repeating phrases
    do_sample=True,           # Enable sampling (not greedy decoding)
    early_stopping=True
)

# Decode and print output
print(processor.decode(outputs[0], skip_special_tokens=True))
