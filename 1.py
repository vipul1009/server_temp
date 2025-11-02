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

# --- 3. Define Your Data and New Batch Function ---

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

def verbalize_batch(triples_string, model, tokenizer):
    """
    Verbalizes a batch of triples (as a single string)
    using the advanced few-shot chat template.
    """
    
    # 1. Create the chat list with your detailed prompt
    chat = [
        { 
          "role": "user", 
          "content": """You are an expert in transforming knowledge base triples into natural, human-friendly sentences.
Your task is to convert all input triples into clear, grammatically correct, and conversational sentences.
Output one sentence per triple, separated by newlines.

**Chain of Thought**:
1. **Parse All Triples**: Identify Subject, Predicate, and Object.
2. **Interpret Predicates**: Use natural phrases (e.g., "is located in," "was born in," "is a," "has a runway length of").
3. **Construct Sentences**: Ensure grammatical agreement and a conversational tone.
4. **Format Output**: One sentence per triple, newline-separated. No headers or commentary.

I will provide you with an example."""
        },
        { 
          "role": "user", 
          "content": """**Example Input**:
Binignit|main ingredients|Sweet potato
Binignit|course|Dessert
Allama Iqbal International Airport|location|Pakistan
Allama Iqbal International Airport|runway length|2900.0"""
          # Note: I changed the example input to use "|" to match your data.
        },
        { 
          "role": "model", 
          "content": """**Example Output**:
The main ingredient of Binignit is Sweet potato.
Binignit is a dessert.
Allama Iqbal International Airport is located in Pakistan.
The runway length of Allama Iqbal International Airport is 2900 meters."""
        },
        { 
          "role": "user", 
          "content": f"""Excellent. Now apply these rules to verbalize the following triples:
{triples_string}"""
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
        max_new_tokens=512,      # Increased token limit for 9 sentences
        do_sample=False,         # Greedy decoding for factual output
    )
    
    # 5. Decode *only* the newly generated tokens
    response_tokens = output_ids[0][prompt_length:]
    text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    return text.strip()

# --- 4. Run the Code ---

print("\n--- Step 2: Starting Batch Verbalization ---")

# Convert your list of triples into a single string
triples_to_verbalize = "\n".join(triples)

print(f"\nInput Batch:\n{triples_to_verbalize}")

# Pass the loaded model and tokenizer to the function
output_text = verbalize_batch(triples_to_verbalize, model, tokenizer)

print(f"\n--- Model Output ---")
print(output_text)

print("\n--- Verbalization Complete ---")
