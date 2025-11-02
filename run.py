#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U transformers')


# ## Local Inference on GPU 
# Model page: https://huggingface.co/openlm-research/open_llama_7b_v2
# 
# ⚠️ If the generated code snippets do not work, please open an issue on either the [model repo](https://huggingface.co/openlm-research/open_llama_7b_v2)
# 			and/or on [huggingface.js](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries-snippets.ts) 🙏

# In[2]:


# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="openlm-research/open_llama_7b_v2")


# In[3]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b_v2")
model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_7b_v2")


# In[4]:


import torch
from transformers import LlamaTokenizer, LlamaForCausalLM


## v2 models
model_path = 'openlm-research/open_llama_7b_v2'


tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)

prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=32
)
print(tokenizer.decode(generation_output[0]))


# In[11]:


import re
from transformers import GenerationConfig

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

prompt_template = """Convert the triple below into a natural English sentence.

Example:
Input: Casablanca|directed_by|Michael Curtiz
Output: Michael Curtiz directed Casablanca.

Now verbalize this triple:
{0}
Output:"""

def verbalize_one(triple):
    prompt = prompt_template.format(triple)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=64,
        temperature=0.4,
        top_p=0.9,
        do_sample=False
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Strip the prompt portion
    text = text.replace(prompt, "").strip()
    # Take first sentence
    text = re.split(r"[\n\r]", text)[0].strip()
    # Ensure punctuation
    if not text.endswith(('.', '!', '?')):
        text += '.'
    return text

for t in triples:
    print(verbalize_one(t))


# In[40]:


import re
from transformers import GenerationConfig

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

prompt_template = """Convert the triple below into a natural English sentence.

Example:
Input: Casablanca|directed_by|Michael Curtiz
Output:Casablanca is directed by Michael Curtiz.

Now verbalize this triple:
{0}
Output:"""

def verbalize_one(triple):
    prompt = prompt_template.format(triple)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=64,
        temperature=0.4,
        top_p=0.9,
        do_sample=False
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Strip the prompt portion
    text = text.replace(prompt, "").strip()
    # Take first sentence
    text = re.split(r"[\n\r]", text)[0].strip()
    # Ensure punctuation
    if not text.endswith(('.', '!', '?')):
        text += '.'
    return text

for t in triples:
    print(verbalize_one(t))

