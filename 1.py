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
    "Kismet|starred_actors|Edward Arnold",
    "Kismet|starred_actors|Ronald Colman",
    "Kismet|starred_actors|James Craig",
    "Kismet|release_year|1944",
    "Kismet|in_language|English",
    "Kismet|has_tags|bd-r",
    "Flags of Our Fathers|directed_by|Clint Eastwood",
    "Flags of Our Fathers|written_by|Paul Haggis",
    "Flags of Our Fathers|written_by|Ron Powers",
    "Flags of Our Fathers|written_by|James Bradley",
    "Flags of Our Fathers|release_year|2006",
    "Flags of Our Fathers|has_genre|War",
    "Flags of Our Fathers|has_imdb_votes|famous",
    "Flags of Our Fathers|has_tags|world war ii",
    "Flags of Our Fathers|has_tags|war",
    "Flags of Our Fathers|has_tags|r",
    "Flags of Our Fathers|has_tags|clint eastwood",
    "Flags of Our Fathers|has_tags|american",
    "Flags of Our Fathers|has_tags|iwo jima",
    "Flags of Our Fathers|has_tags|flag",
    "The Bride Wore Black|directed_by|François Truffaut",
    "The Bride Wore Black|written_by|Cornell Woolrich",
    "The Bride Wore Black|written_by|François Truffaut",
    "The Bride Wore Black|starred_actors|Jeanne Moreau",
    "The Bride Wore Black|starred_actors|Michel Bouquet",
    "The Bride Wore Black|starred_actors|Charles Denner",
    "The Bride Wore Black|release_year|1968",
    "The Bride Wore Black|in_language|French",
    "The Bride Wore Black|has_tags|bd-r",
    "The Bride Wore Black|has_tags|revenge",
    "The Bride Wore Black|has_tags|wedding",
    "The Bride Wore Black|has_tags|françois truffaut",
    "The Bride Wore Black|has_tags|black",
    "The Bride Wore Black|has_tags|bride",
    "Dirty Filthy Love|directed_by|Adrian Shergold",
    "Dirty Filthy Love|written_by|Jeff Pope",
    "Dirty Filthy Love|starred_actors|Michael Sheen",
    "Dirty Filthy Love|starred_actors|Claudie Blakley",
    "Dirty Filthy Love|starred_actors|Anastasia Griffith",
    "Dirty Filthy Love|starred_actors|Adrian Bower",
    "Dirty Filthy Love|release_year|2004",
    "Dirty Filthy Love|has_genre|Drama",
    "The Dark Horse|directed_by|Alfred E. Green",
    "The Dark Horse|starred_actors|Bette Davis",
    "The Dark Horse|starred_actors|Warren William",
    "The Dark Horse|release_year|1932",
    "The Dark Horse|has_genre|Comedy",
    "The Dark Horse|has_tags|alfred e. green",
    "The Sentinel|directed_by|Clark Johnson",
    "The Sentinel|written_by|Gerald Petievich",
    "The Sentinel|starred_actors|Michael Douglas",
    "The Sentinel|starred_actors|Kiefer Sutherland",
    "The Sentinel|starred_actors|Eva Longoria",
    "The Sentinel|release_year|2006",
    "The Sentinel|has_genre|Thriller",
    "The Sentinel|has_genre|Crime",
    "The Sentinel|has_tags|thriller",
    "The Sentinel|has_tags|crime",
    "The Sentinel|has_tags|michael douglas",
    "The Sentinel|has_tags|secret service",
    "Funny About Love|directed_by|Leonard Nimoy",
    "Funny About Love|written_by|Norman Steinberg",
    "Funny About Love|written_by|David Frankel",
    "Funny About Love|written_by|Bob Greene",
    "Funny About Love|starred_actors|Gene Wilder",
    "Funny About Love|release_year|1990",
    "Funny About Love|has_genre|Comedy",
    "Kissed|directed_by|Lynne Stopkewich",
    "Kissed|written_by|Barbara Gowdy",
    "Kissed|written_by|Lynne Stopkewich",
    "Kissed|starred_actors|Molly Parker",
    "Kissed|starred_actors|Peter Outerbridge",
    "Kissed|release_year|1996",
    "Kissed|has_tags|necrophilia",
    "Alive|written_by|Tsutomu Takahashi",
    "Alive|starred_actors|Hideo Sakaki",
    "Alive|release_year|2002",
    "Alive|in_language|Japanese",
    "Alive|has_genre|Action",
    "Alive|has_tags|japan",
    "Alive|has_tags|prison",
    "Snow Queen|directed_by|David Wu",
    "Snow Queen|starred_actors|Bridget Fonda",
    "Snow Queen|starred_actors|Chelsea Hobbs",
    "Snow Queen|release_year|2002",
    "Chopper|directed_by|Andrew Dominik",
    "Chopper|written_by|Andrew Dominik",
    "Chopper|starred_actors|Eric Bana",
    "Chopper|starred_actors|David Field",
    "Chopper|starred_actors|Simon Lyndon",
    "Chopper|release_year|2000",
    "Chopper|has_tags|australia",
    "Chopper|has_tags|australian",
    "Chopper|has_tags|eric bana",
    "Chopper|has_tags|andrew dominik",
    "The Mistress of Spices|directed_by|Paul Mayeda Berges",
    "The Mistress of Spices|written_by|Gurinder Chadha",
    "The Mistress of Spices|written_by|Paul Mayeda Berges",
    "The Mistress of Spices|written_by|Chitra Banerjee Divakaruni",
    "The Mistress of Spices|release_year|2005"
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

    # --- 5. Read File and Generate Summary ---
    
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

    # 6. Clean and save the final summary
    summary_filename = "summary.txt"
    cleaned_summary = summary_text.strip()

    print("\n--- Generated Summary ---")
    print(cleaned_summary) # Display the summary on the console

    # --- MODIFICATION: Save the summary to summary.txt ---
    with open(summary_filename, "w", encoding="utf-8") as f:
        f.write(cleaned_summary)
    
    print(f"\n--- Summary successfully saved to {summary_filename} ---")


except Exception as e:
    print(f"\nAn error occurred: {e}")

print("\n--- Full Process Complete ---")
