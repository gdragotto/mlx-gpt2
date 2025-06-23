import mlx.core as mx
import mlx.utils as utils
import json
from model import GPT

def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    # Load tokenizer info
    with open("data/tokenizer.json", "r") as f:
        tokenizer_info = json.load(f)
    
    vocab_size = tokenizer_info["vocab_size"]
    itos = tokenizer_info["itos"]
    stoi = tokenizer_info["stoi"]
    
    # Create decode function
    decode = lambda x: ''.join([itos[str(i)] for i in x])
    
    # Create encode function
    encode = lambda x: [stoi[c] for c in x]
    
    # Load model
    model = GPT(vocab_size)
    state_dict = utils.load("data/gpt2_mlx_model.safetensors")
    model = model.update(state_dict)
    
    return model, decode, encode

def generate_text(model, decode, encode, prompt, max_new_tokens=1000):
    """Generate text using the trained model with a given prompt"""
    # Encode the prompt
    prompt_tokens = encode(prompt)
    
    # Generate continuation
    completion_tokens = model.generate(mx.array([prompt_tokens]), max_new_tokens)[0].tolist()
    
    # Decode the full text (prompt + completion)
    full_text = decode(completion_tokens)
    return full_text

if __name__ == "__main__":
    # Load model and tokenizer
    model, decode, encode = load_model_and_tokenizer()
    
    # Get user input
    print("Enter your prompt text (press Enter when done):")
    user_input = input().strip()
    
    if not user_input:
        print("No input provided. Using default prompt.")
        user_input = "The quick brown fox"
    
    print(f"\nGenerating text starting with: '{user_input}'")
    
    # Generate text
    completion = generate_text(model, decode, encode, user_input, max_new_tokens=1000)
    print("\nGenerated text:")
    print(completion)