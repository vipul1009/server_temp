from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load GPT-OSS 3B model from Hugging Face (replace with actual model name if needed)
model_name = "EleutherAI/gpt-neo-2.7B"  # Placeholder name; update if different

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define the prompt
prompt = "Verbalize the triple: Trump | President | America"

# Generate output
output = generator(prompt, max_length=50, do_sample=True, temperature=0.7)

# Print result
print("Generated sentence:", output[0]['generated_text'])
