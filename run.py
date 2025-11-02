import re
import torch
from transformers import GenerationConfig # Note: This import isn't used in your func, which is fine!

# --- Your data ---
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

# --- CORRECTED verbalization function ---

def verbalize_one(triple):
    # 1. Format the prompt using the model's required chat template
    # We provide the "one-shot" example as conversation history
    chat = [
        { "role": "user", "content": "Convert the triple below into a natural English sentence.\nInput: Casablanca|directed_by|Michael Curtiz" },
        { "role": "model", "content": "Casablanca is directed by Michael Curtiz." },
        { "role": "user", "content": f"Now verbalize this triple:\n{triple}" }
    ]
    
    # 2. Tokenize the entire chat
    # We let the tokenizer handle all special tokens and formatting
    input_ids = tokenizer.apply_chat_template(
        chat,
        return_tensors="pt",
        add_generation_prompt=True # This adds the '<start_of_turn>model' for us
    ).to(model.device)

    # 3. Store the length of our prompt
    prompt_length = input_ids.shape[1]

    # 4. Generate the response
    output_ids = model.generate(
        input_ids,
        max_new_tokens=64,
        do_sample=False  # Use greedy decoding for factual answers
    )
    
    # 5. Decode *only* the newly generated tokens
    # This is much more reliable than text.replace()
    response_tokens = output_ids[0][prompt_length:]
    text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    # 6. Your post-processing (this was good!)
    text = re.split(r"[\n\r]", text)[0].strip()
    if not text.endswith(('.', '!', '?')):
        text += '.'
    return text

# --- Run the loop ---
print("--- Starting Triple Verbalization ---")
for t in triples:
    print(f"\nInput: {t}")
    output_text = verbalize_one(t)
    print(f"Output: {output_text}")

print("\n--- Verbalization Complete ---")
