from transformers import AutoProcessor, AutoModelForImageTextToText

# Load model and processor
processor = AutoProcessor.from_pretrained("google/gemma-3-27b-it")
model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-27b-it")

# Input message (must be in message format for this model)
messages = [
    {"role": "user", "content": [{"type": "text", "text": "What is SVM in machine learning?"}]}
]

# Process input properly
inputs = processor(
    messages=messages,
    return_tensors="pt"
).to(model.device)

# Generate with advanced parameters
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    do_sample=True,
    early_stopping=True
)

# Decode output
print(processor.decode(outputs[0], skip_special_tokens=True))
