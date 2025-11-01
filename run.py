# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("google/gemma-3-27b-it")
model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-27b-it")

# Modified messages for text-only input
messages = [
    {
        "role": "user",
        "content": [
            # The image dictionary has been removed
            {"type": "text", "text": "WHAT ARE SVM IN MACHINE LEARNING"} # The text prompt has been changed
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# Increased max_new_tokens to allow for a more complete answer
outputs = model.generate(**inputs, max_new_tokens=256) 
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
