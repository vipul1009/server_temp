import time
import logging
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    "MODEL_NAME": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "MAX_NEW_TOKENS": 4096,
    "TEMPERATURE": 0.7,
    "TOP_P": 0.9,
    "TOP_K": 50,
    "SLEEP_INTERVAL": 0,
    "CHUNK_SIZE": 50
}

# ---------------- PROMPTS ----------------
prompt_verbalize = """
You are an expert in transforming knowledge base triples into natural, human-friendly sentences for entities like foods, people, airports, and regions. The input is a list of triples in the format: Subject Predicate Object, all provided at once. Your task is to convert all triples into clear, grammatically correct, and conversational sentences that reflect the predicate's semantic role, ensuring conciseness and natural phrasing for a general audience. Output one sentence per triple, separated by newlines.

**Chain of Thought**:
1. **Parse All Triples**:
   - For each triple, identify Subject (e.g., "Binignit"), Predicate (e.g., "main ingredients"), and Object (e.g., "Sweet potato").
   - Note the Predicate's semantic role (e.g., 'main ingredients' for components, 'birth place' for origin).
2. **Interpret Predicates**:
   - Use phrases like "The main ingredient(s) of [Subject] is/are [Object]" for 'main ingredients'.
   - Use "is located in" for 'location', "was born in" for 'birth place', "is" for 'course' or 'type', etc.
   - For numerical predicates (e.g., 'runway length'), include units (e.g., "meters").
   - For lists (e.g., "Ground almond, jam"), join with "and" (e.g., "ground almond and jam").
3. **Construct Sentences**:
   - Ensure grammatical agreement and correct tense (present for states, past for events).
   - Use conversational tone, avoiding jargon (e.g., "is assembled in" not "has assembly location").
4. **Format Output**:
   - One sentence per triple, separated by newlines.
   - Use exact Subject and Object text, adjusting only for grammar if needed.
   - No additional commentary or headers.

**Examples**:
Input:
Binignit main ingredients Sweet potato
Binignit course Dessert
Allama Iqbal International Airport location Pakistan
Allama Iqbal International Airport runway length 2900.0
Output:
The main ingredient of Binignit is Sweet potato.
Binignit is a dessert.
Allama Iqbal International Airport is located in Pakistan.
The runway length of Allama Iqbal International Airport is 2900 meters.

**Instructions**:
- Process all triples together, producing one sentence per triple.
- Use natural phrasing, correct tense, and implied units (e.g., meters for runway length).
- Output only sentences, one per line.

Verbalize the following triples:
{0}

"""

prompt_summarize = """
You are an expert at summarizing verbose, fact-rich texts about entities—such as people, films, places, foods, objects, or events—into a single, highly detailed, cohesive paragraph. Your task is to produce a paragraph that fully captures every fact in the input, including attributes, relationships, roles, creators, actors, release dates, genres, locations, languages, historical events, ingredients, and cultural significance. The summary must eliminate redundancy, compress verbose descriptions, avoid trivial details, and prevent list-like or comma-heavy structures, while remaining fully readable and natural. Use varied sentence structures, conjunctions, prepositions, and relative clauses (e.g., "which," "where," "and") to integrate facts smoothly. Use present tense for ongoing states or current facts and past tense for historical events or completed actions. Adjust paragraph length according to the input's complexity, ensuring exhaustive coverage of all entities while keeping it concise and fluent.

Instructions: Summarize the following input into one dense, cohesive paragraph. Integrate all entities and include every fact: attributes, relationships, roles, creators, actors, release dates, genres, languages, locations, events, and cultural significance. Avoid repetition, eliminate trivial or redundant details, and prevent list-like structures. Use natural sentence flow with varied syntax, conjunctions, prepositions, and relative clauses, maintaining clarity, readability, and a proportionate paragraph length reflecting the input's complexity.

{0}
"""

prompt_qa = """
You are a precise question-answering assistant specialized in multi-hop reasoning over structured summaries. Your sole task is to answer the provided questions using EXCLUSIVELY the information contained in the given summary. You MUST NOT use any external knowledge, general knowledge, prior training data, or anything beyond the exact text of the summary. If a piece of information required to answer a question is not explicitly stated or directly inferable from the summary's text, you MUST output "NOT PRESENT" for that question. Do not hallucinate, assume, extrapolate, or fill gaps; every inference must be based on stated facts.

Key rules:
- Base every answer ONLY on the summary text.
- For multi-hop reasoning: Trace explicit chains only (e.g., A → B via direct relation in summary, B → C via another relation).
- Answers must be concise: one word/phrase for single items, or multiple items separated by | (e.g., "Top Hat|Kitty Foyle|The Barkleys of Broadway").
- If no relevant information in the summary, output exactly "NOT PRESENT".
- No explanations, no additional text—only the numbered answers.

Summary:
{summarized_text}

Questions:
{questions_list}
"""

# ---------------- MODEL CONFIGURATION ----------------
def initialize_model():
    """Initialize TinyLlama model and tokenizer."""
    logger.info("=" * 50)
    logger.info("Loading TinyLlama model...")
    logger.info("=" * 50)
    
    try:
        # Check if GPU is available
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info("✓ Using GPU for inference")
        else:
            logger.info("⚠ Using CPU for inference (this will be slower)")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_NAME"])
        
        # Load model
        logger.info("Loading model (this may take a few minutes)...")
        if use_gpu:
            # GPU loading with device_map
            model = AutoModelForCausalLM.from_pretrained(
                CONFIG["MODEL_NAME"],
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline WITHOUT device parameter when using device_map
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer
            )
        else:
            # CPU loading
            model = AutoModelForCausalLM.from_pretrained(
                CONFIG["MODEL_NAME"],
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline with device=-1 for CPU
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=-1
            )
        
        logger.info("✓ Model loaded successfully!")
        logger.info("=" * 50)
        return generator
        
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        logger.error("Make sure you have enough memory and the model can be downloaded.")
        sys.exit(1)


def generate_text(generator, prompt):
    """Generate text using TinyLlama model."""
    try:
        # Format prompt for chat model
        formatted_prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        response = generator(
            formatted_prompt,
            max_new_tokens=CONFIG["MAX_NEW_TOKENS"],
            temperature=CONFIG["TEMPERATURE"],
            top_p=CONFIG["TOP_P"],
            top_k=CONFIG["TOP_K"],
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id,
            return_full_text=False,
            num_return_sequences=1
        )
        
        generated_text = response[0]['generated_text'].strip()
        return generated_text
        
    except Exception as e:
        logger.error(f"✗ Error in generate_text: {e}")
        return ""


def read_txt_in_chunks(file_path, chunk_size=None):
    """Read a .txt file and yield chunks of lines."""
    if chunk_size is None:
        chunk_size = CONFIG["CHUNK_SIZE"]
    
    if not os.path.exists(file_path):
        logger.error(f"✗ Input file not found: {file_path}")
        sys.exit(1)
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    logger.info(f"✓ Loaded {len(lines)} lines from {file_path}")
    
    for i in range(0, len(lines), chunk_size):
        yield lines[i:i + chunk_size]


def read_first_20_questions(questions_file):
    """Read and return the first 20 non-empty lines from questions.txt"""
    if not os.path.exists(questions_file):
        logger.warning(f"⚠ Questions file '{questions_file}' not found.")
        return []
    
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    
    logger.info(f"✓ Loaded {min(len(questions), 20)} questions from {questions_file}")
    return questions[:20]


# ---------------- MAIN PIPELINE ----------------
def process_txt_file(input_txt, output_dir, questions_file):
    """Process input text and perform summarization + QA using first 20 questions."""
    logger.info("=" * 50)
    logger.info("Starting processing pipeline")
    logger.info("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"✓ Output directory: {output_dir}")

    verbalized_file = os.path.join(output_dir, "verbalized_all.txt")
    summarized_file = os.path.join(output_dir, "summarized_all.txt")
    qa_file = os.path.join(output_dir, "qa_answers_first20.txt")

    # Empty the files initially
    for file in [verbalized_file, summarized_file, qa_file]:
        with open(file, "w", encoding="utf-8") as f:
            f.write("")
    logger.info("✓ Output files initialized")

    # Initialize model once
    generator = initialize_model()

    # Count total chunks
    total_chunks = sum(1 for _ in read_txt_in_chunks(input_txt))
    logger.info(f"✓ Total chunks to process: {total_chunks}")
    logger.info("=" * 50)

    for chunk_idx, chunk in enumerate(read_txt_in_chunks(input_txt), start=1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing chunk {chunk_idx}/{total_chunks} ({len(chunk)} lines)")
        logger.info(f"{'='*50}")

        try:
            # Step 1: Verbalize
            logger.info("Step 1/4: Verbalizing triples...")
            prompt_v = prompt_verbalize.format("\n".join(chunk))
            verbalized_text = generate_text(generator, prompt_v)
            
            if not verbalized_text:
                logger.warning(f"⚠ Empty verbalization for chunk {chunk_idx}")
                continue
            logger.info(f"✓ Verbalization complete ({len(verbalized_text)} chars)")

            # Step 2: Summarize
            logger.info("Step 2/4: Summarizing text...")
            prompt_s = prompt_summarize.format(verbalized_text)
            summarized_text = generate_text(generator, prompt_s)
            
            if not summarized_text:
                logger.warning(f"⚠ Empty summary for chunk {chunk_idx}")
                continue
            logger.info(f"✓ Summarization complete ({len(summarized_text)} chars)")

            # Step 3: Load first 20 questions from questions.txt
            logger.info("Step 3/4: Loading questions...")
            first20_questions = read_first_20_questions(questions_file)
            if not first20_questions:
                logger.warning("⚠ No questions found - skipping QA.")
                continue

            questions_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(first20_questions))

            # Step 4: QA
            logger.info("Step 4/4: Answering questions...")
            prompt_q = prompt_qa.format(
                summarized_text=summarized_text,
                questions_list=questions_block
            )
            qa_answers = generate_text(generator, prompt_q)
            logger.info(f"✓ QA complete ({len(qa_answers)} chars)")

            # Step 5: Save outputs
            with open(verbalized_file, "a", encoding="utf-8") as f:
                f.write(verbalized_text + "\n\n")

            with open(summarized_file, "a", encoding="utf-8") as f:
                f.write(summarized_text + "\n\n")

            with open(qa_file, "a", encoding="utf-8") as f:
                f.write(f"--- CHUNK {chunk_idx} ---\n")
                f.write(qa_answers + "\n\n")

            logger.info(f"✓ Chunk {chunk_idx} saved successfully")
            
            if CONFIG["SLEEP_INTERVAL"] > 0:
                time.sleep(CONFIG["SLEEP_INTERVAL"])
        
        except Exception as e:
            logger.error(f"✗ Error processing chunk {chunk_idx}: {e}")
            continue

    logger.info("\n" + "=" * 50)
    logger.info("✅ All chunks processed successfully!")
    logger.info(f"✓ Results saved in: {output_dir}")
    logger.info("=" * 50)


# ---------------- RUN ----------------
if __name__ == "__main__":
    # File paths
    input_txt = "output2.txt"
    questions_file = "questionall.txt"
    output_dir = "output41_txt"
    
    # Check if input files exist
    if not os.path.exists(input_txt):
        logger.error(f"✗ Input file not found: {input_txt}")
        logger.info("Please make sure the input file exists.")
        sys.exit(1)
    
    if not os.path.exists(questions_file):
        logger.warning(f"⚠ Questions file not found: {questions_file}")
        logger.info("QA step will be skipped.")
    
    # Run the pipeline
    process_txt_file(input_txt, output_dir, questions_file)
