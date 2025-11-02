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
triple = [
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
    "The Mistress of Spices|release_year|2005",
    "Snow Dogs|directed_by|Brian Levant",
    "Snow Dogs|written_by|Gary Paulsen",
    "Snow Dogs|starred_actors|James Coburn",
    "Snow Dogs|release_year|2002",
    "Snow Dogs|has_genre|Comedy",
    "Snow Dogs|has_tags|brian levant",
    "House|directed_by|Steve Miner",
    "House|starred_actors|William Katt",
    "House|starred_actors|George Wendt",
    "House|starred_actors|Kay Lenz",
    "House|starred_actors|Richard Moll",
    "House|release_year|1986",
    "House|has_genre|Comedy",
    "House|has_genre|Horror",
    "The Hidden Fortress|directed_by|Akira Kurosawa",
    "The Hidden Fortress|written_by|Akira Kurosawa",
    "The Hidden Fortress|release_year|1958",
    "The Hidden Fortress|has_tags|akira kurosawa",
    "The Hidden Fortress|has_tags|toshiro mifune",
    "The Hidden Fortress|has_tags|kurosawa",
    "The Hidden Fortress|has_tags|princess",
    "I Want to Live!|directed_by|Robert Wise",
    "I Want to Live!|written_by|Nelson Gidding",
    "I Want to Live!|written_by|Don Mankiewicz",
    "I Want to Live!|written_by|Barbara Graham",
    "I Want to Live!|written_by|Ed Montgomery",
    "I Want to Live!|starred_actors|Susan Hayward",
    "I Want to Live!|starred_actors|Theodore Bikel",
    "I Want to Live!|starred_actors|Simon Oakland",
    "I Want to Live!|release_year|1958",
    "I Want to Live!|has_tags|murder",
    "I Want to Live!|has_tags|robert wise",
    "Mrs. Parkington|directed_by|Tay Garnett",
    "Mrs. Parkington|written_by|Robert Thoeren",
    "Mrs. Parkington|written_by|Louis Bromfield",
    "Mrs. Parkington|written_by|Polly James",
    "Mrs. Parkington|starred_actors|Walter Pidgeon",
    "Mrs. Parkington|starred_actors|Greer Garson",
    "Mrs. Parkington|release_year|1944",
    "Mrs. Parkington|has_genre|Drama",
    "Mrs. Parkington|has_tags|tay garnett",
    "The Wolf Man|directed_by|George Waggner",
    "The Wolf Man|written_by|Curt Siodmak",
    "The Wolf Man|starred_actors|Claude Rains",
    "The Wolf Man|starred_actors|Ralph Bellamy",
    "The Wolf Man|release_year|1941",
    "The Wolf Man|has_genre|Drama",
    "The Wolf Man|has_genre|Horror",
    "The Wolf Man|has_tags|werewolf",
    "A Cruel Romance|directed_by|Eldar Ryazanov",
    "A Cruel Romance|written_by|Eldar Ryazanov",
    "A Cruel Romance|release_year|1984",
    "A Cruel Romance|in_language|Russian",
    "A Cruel Romance|has_genre|Drama",
    "A Cruel Romance|has_genre|Romance",
    "A Cruel Romance|has_tags|russian",
    "Love Happens|directed_by|Brandon Camp",
    "Love Happens|written_by|Brandon Camp",
    "Love Happens|starred_actors|Aaron Eckhart",
    "Love Happens|starred_actors|Jennifer Aniston",
    "Love Happens|release_year|2009",
    "Love Happens|has_genre|Drama",
    "Love Happens|has_genre|Romance",
    "Love Happens|has_tags|jennifer aniston"
]

def verbalize_batch(triples_string, model, tokenizer):
    """
    Verbalizes a batch of triples (as a single string)
    using the advanced few-shot chat template.
    """
    
    # 1. Create the chat list with your detailed prompt
    # 1. Format the prompt using the model's required chat template.
    # This "few-shot" prompt provides a role and multiple examples
    # to teach the model how to handle different predicates.
    
    chat = [
        { 
          "role": "user", 
          "content": """You are an expert assistant for natural language generation. Your task is to convert a single knowledge base triple (Subject|Predicate|Object) into one clear, natural-sounding English sentence.

I will provide you with several examples so you can learn the correct style."""
        },
        { 
          "role": "model", 
          "content": "Understood. I am ready for the examples." 
        },
        { 
          "role": "user", 
          "content": "Input: Kismet|directed_by|William Dieterle" 
        },
        { 
          "role": "model", 
          "content": "Kismet was directed by William Dieterle." 
        },
        { 
          "role": "user", 
          "content": "Input: The Dark Horse|release_year|1932" 
        },
        { 
          "role": "model", 
          "content": "The Dark Horse was released in 1932." 
        },
        { 
          "role": "user", 
          "content": "Input: Flags of Our Fathers|has_genre|War" 
        },
        { 
          "role": "model", 
          "content": "Flags of Our Fathers is a War film." 
        },
        { 
          "role": "user", 
          "content": "Input: Kismet|starred_actors|Marlene Dietrich" 
        },
        { 
          "role": "model", 
          "content": "Kismet starred Marlene Dietrich." 
        },
        { 
          "role": "user", 
          "content": f"""Excellent. You have learned the patterns. Now, apply the same logic to the following triple.
Output only the single, verbalized sentence.

Input: {triple}"""
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
