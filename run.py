import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_gemma_8bit():
    """
    Load Gemma with 8-bit quantization for reduced memory usage
    """
    model_id = "google/gemma-2-27b-it"
    
    # Configure 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
    
    print("Loading model with 8-bit quantization...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
    )
    
    print("Model loaded!\n")
    
    # Interactive chat
    chat_history = []
    
    print("Chat with Gemma 3 27B (type 'quit' to exit)\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        chat_history.append({"role": "user", "content": user_input})
        
        prompt = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        assistant_response = response.split("<start_of_turn>model\n")[-1]
        
        print(f"\nGemma: {assistant_response}\n")
        
        chat_history.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    load_gemma_8bit()
