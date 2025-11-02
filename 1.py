# --- 1. Installation ---
#
# First, you must run this in your terminal to install the required libraries.
# Your CUDA 12.1 is perfect for this.
#
# pip install torch transformers accelerate bitsandbytes
#
# ----------------------------------------

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

print("\n--- Model Loaded Successfully on GPU! ---")

# --- 3. Define Your Data and Function ---

# Your list of triples
triples = [
    "Kismet|directed_by|William Dieterle",
    "Kismet|written_by|Edward Knoblock",
    "Kismet|starred_actors|Marlene Dietrich",
    "Flags of Our Fathers|directed_by|Clint Eastwood",
    "Flags of Our Fathers|written_by|Paul Haggis",
    "Flags of Our Fathers|has_genre|War",
    "The Dark Horse|directed_by|Alfred E. Green",
    "The Dark Horse|starred_actors|Bette Davis",
    "The Dark Horse|release_year|1932"
]

def verbalize_one(triple, model, tokenizer):
    """
    Verbalizes a single triple using the model's chat template.
    """
    
    # 1. Format the prompt using the model's required chat template
    # We provide the "one-shot" example as conversation history
    chat = [
        { "role": "user", "content": "Convert the triple below into a natural English sentence.\nInput: Casablanca|directed_by|Michael Curtiz" },
        { "role": "model", "content": "Casablanca is directed by Michael Curtiz." },
        { "role": "user", "content": f"Now verbalize this triple:\n{triple}" }
    ]
    
    # 2. Tokenize the entire chat
    input_ids = tokenizer.apply_chat_template(
        chat,
        return_tensors="pt",
        add_generation_prompt=True # Adds the '<start_of_turn>model' token
    ).to(model.device)

    # 3. Store the length of our prompt to slice it out later
    prompt_length = input_ids.shape[1]

    # 4. Generate the response
    output_ids = model.generate(
        input_ids,
        max_new_tokens=64,     # Max length of the *new* sentence
        do_sample=False,       # Use greedy decoding (best for this factual task)
    )
    
    # 5. Decode *only* the newly generated tokens
    # This is much more reliable than trying to strip the prompt text
    response_tokens = output_ids[0][prompt_length:]
    text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    # 6. Post-process the output
    text = re.split(r"[\n\r]", text)[0].strip() # Take first line
    if not text.endswith(('.', '!', '?')):
        text += '.'
    return text

# --- 4. Run the Verbalization and Save to File ---

print("\n--- Step 2: Starting Triple Verbalization ---")
output_filename = "verbalized.txt"

# Open the file in 'w' (write) mode to save the outputs
try:
    with open(output_filename, "w", encoding="utf-8") as f:
        for t in triples:
            print(f"\nInput: {t}")
            # Pass the loaded model and tokenizer to the function
            output_text = verbalize_one(t, model, tokenizer)
            print(f"Output: {output_text}")
            
            # --- MODIFICATION: Write sentence to file ---
            f.write(output_text + "\n")

    print(f"\n--- Verbalization Complete. Sentences saved to {output_filename} ---")

    # --- 5. NEW: Read File and Generate Summary ---
    
    print(f"\n--- Step 3: Reading {output_filename} for Summarization ---")
    
    # Read the content of the file we just created
    with open(output_filename, "r", encoding="utf-8") as f:
        text_to_summarize = f.read()

    print("--- File Content Read Successfully: ---")
    print(text_to_summarize)
    print("---------------------------------------")

    print("\n--- Step 4: Generating Summary ---")

    # 1. Format the summarization prompt
    summary_chat = [
        { "role": "user", "content": f"Summarize the following movie descriptions into a single, cohesive paragraph:\n\n{text_to_summarize}" }
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
        max_new_tokens=256,   # Allow more tokens for a summary
        do_sample=True,       # Enable sampling for a more creative summary
        temperature=0.7,
        top_p=0.9,
    )

    # 5. Decode *only* the newly generated tokens
    summary_response_tokens = summary_output_ids[0][summary_prompt_length:]
    summary_text = tokenizer.decode(summary_response_tokens, skip_special_tokens=True)

    # 6. Print the final summary
    print("\n--- Generated Summary ---")
    print(summary_text.strip())


except Exception as e:
    print(f"\nAn error occurred: {e}")

print("\n--- Full Process Complete ---")
