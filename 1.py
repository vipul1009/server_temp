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
    {
        "role": "user",
        "content": (
            "You are an expert in transforming knowledge base triples into natural, human-friendly sentences "
            "for entities like foods, people, airports, and regions. The input is a list of triples in the format: "
            "Subject Predicate Object, all provided at once. Your task is to convert all triples into clear, "
            "grammatically correct, and conversational sentences that reflect the predicate's semantic role, "
            "ensuring conciseness and natural phrasing for a general audience. Output one sentence per triple, "
            "separated by newlines.\n\n"
            "**Chain of Thought**:\n"
            "1. **Parse All Triples**:\n"
            "   - For each triple, identify Subject (e.g., 'Binignit'), Predicate (e.g., 'main ingredients'), "
            "and Object (e.g., 'Sweet potato').\n"
            "   - Note the Predicate's semantic role (e.g., 'main ingredients' for components, 'birth place' for origin).\n"
            "2. **Interpret Predicates**:\n"
            "   - Use phrases like 'The main ingredient(s) of [Subject] is/are [Object]' for 'main ingredients'.\n"
            "   - Use 'is located in' for 'location', 'was born in' for 'birth place', 'is' for 'course' or 'type', etc.\n"
            "   - For numerical predicates (e.g., 'runway length'), include units (e.g., 'meters').\n"
            "   - For lists (e.g., 'Ground almond, jam'), join with 'and' (e.g., 'ground almond and jam').\n"
            "3. **Construct Sentences**:\n"
            "   - Ensure grammatical agreement and correct tense (present for states, past for events).\n"
            "   - Use conversational tone, avoiding jargon (e.g., 'is assembled in' not 'has assembly location').\n"
            "4. **Format Output**:\n"
            "   - One sentence per triple, separated by newlines.\n"
            "   - Use exact Subject and Object text, adjusting only for grammar if needed.\n"
            "   - No additional commentary or headers.\n\n"
            "**Examples**:\n"
            "Input:\n"
            "Casablanca|directed_by|Michael Curtiz\n"
            "Output:\n"
            "Casablanca is directed by Michael Curtiz.\n\n"
            "**Instructions**:\n"
            "- Process all triples together, producing one sentence per triple.\n"
            "- Use natural phrasing, correct tense, and implied units (e.g., meters for runway length).\n"
            "- Output only sentences, one per line."
        )
    },
    {
        "role": "user",
        "content": "Convert the triple below into a natural English sentence.\nInput: Casablanca|directed_by|Michael Curtiz"
    },
    {
        "role": "assistant",
        "content": "Casablanca is directed by Michael Curtiz."
    },
    {
        "role": "user",
        "content": f"Now verbalize these triples:\n{triple}"
    }
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

# --- 4. Run the Code ---

print("\n--- Step 2: Starting Triple Verbalization ---")
for t in triples:
    print(f"\nInput: {t}")
    # Pass the loaded model and tokenizer to the function
    output_text = verbalize_one(t, model, tokenizer)
    print(f"Output: {output_text}")

print("\n--- Verbalization Complete ---")
