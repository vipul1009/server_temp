from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

prompt = "Using only the words from the triple, generate a natural sentence: Triple: Trump | President | America"
output = generator(prompt, max_length=50, do_sample=True, temperature=0.7)

print("Generated:", output[0]['generated_text'])
