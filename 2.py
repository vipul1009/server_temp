import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import os # Added for file operations

print("--- Step 1: Loading 4-bit Model (gemma-2-27b-it) ---")

# --- 2. Load the Model and Tokenizer ---

# This is the official instruction-tuned model
model_id = "google/gemma-2-27b-it"

# This tells the transformer to load the model in 4-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16  # Your V100 supports bfloat16 for faster compute
)

# Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the Model
# - `quantization_config` applies the 4-bit loading
# - `device_map="auto"` automatically finds and uses your V100 GPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("\n--- Model Loaded Successfully on GPU! ---")# (Assuming the 'try' block starts here, along with 'tokenizer' and 'model' being defined)
try:
    # --- 5. Read File and Generate Summary ---
    
    # --- MODIFICATION: Using "verbalized.txt" as the filename ---
    print(f"\n--- Step 3: Reading verbalized.txt for Summarization ---")
    
    # Read the content of the file we just created
    with open("verbalized.txt", "r", encoding="utf-8") as f:
        text_to_summarize = f.read()
    # --- END OF MODIFICATION ---

    print("--- File Content Read Successfully: ---")
    print(text_to_summarize)
    print("---------------------------------------")

    print("\n--- Step 4: Generating Summary ---")

    # 1. Format the summarization prompt
    
    # This is the full instruction text from your 'prompt_summarize'
    prompt_text_content = """You are an expert at summarizing verbose, fact-rich texts about entities—such as people, films, places, foods, objects, or events—into a single, highly detailed, cohesive paragraph. Your task is to produce a paragraph that fully captures every fact in the input, including attributes, relationships, roles, creators, actors, release dates, genres, locations, languages, historical events, ingredients, and cultural significance. The summary must eliminate redundancy, compress verbose descriptions, avoid trivial details, and prevent list-like or comma-heavy structures, while remaining fully readable and natural. Use varied sentence structures, conjunctions, prepositions, and relative clauses (e.g., "which," "where," "and") to integrate facts smoothly. Use present tense for ongoing states or current facts and past tense for historical events or completed actions. Adjust paragraph length according to the input's complexity, ensuring exhaustive coverage of all entities while keeping it concise and fluent.

Instructions: Summarize the following input into one dense, cohesive paragraph. Integrate all entities and include every fact: attributes, relationships, roles, creators, actors, release dates, genres, languages, locations, events, and cultural significance. Avoid repetition, eliminate trivial or redundant details, and prevent list-like structures. Use natural sentence flow with varied syntax, conjunctions, prepositions, and relative clauses, maintaining clarity, readability, and a proportionate paragraph length reflecting the input's complexity."""

    # This code builds the 'summary_chat' variable in the format you requested,
    # combining the detailed instructions with the text_to_summarize variable 
    # that was read from the file.
    summary_chat = [
        { 
            "role": "user", 
            "content": f"{prompt_text_content}\n\n{text_to_summarize}"
        }
    ]

    # 2. Tokenize the summarization prompt
    summary_input_ids = tokenizer.apply_chat_template(
        summary_chat,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    # 3. Store the prompt length to slice it out
    summary_prompt_length = summary_input_ids.shape[1]

    # 4. Generate the summary
    # Note: Using do_sample=True for better, more natural summarization
    summary_output_ids = model.generate(
        summary_input_ids,
        max_new_tokens=256,    # Allow more tokens for a summary
        do_sample=True,        # Enable sampling for a more creative summary
        temperature=0.7,
        top_p=0.9,
    )

    # 5. Decode *only* the newly generated tokens
    summary_response_tokens = summary_output_ids[0][summary_prompt_length:]
    summary_text = tokenizer.decode(summary_response_tokens, skip_special_tokens=True)

    # 6. Clean and save the final summary
    summary_filename = "summary.txt"
    cleaned_summary = summary_text.strip()

    print("\n--- Generated Summary ---")
    print(cleaned_summary) # Display the summary on the console

    # Save the summary to summary.txt
    with open(summary_filename, "w", encoding="utf-8") as f:
        f.write(cleaned_summary)
    
    print(f"\n--- Summary successfully saved to {summary_filename} ---")


except Exception as e:
    print(f"\nAn error occurred: {e}")

print("\n--- Full Process Complete ---")
