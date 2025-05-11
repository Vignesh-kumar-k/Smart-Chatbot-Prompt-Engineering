from llama_cpp import Llama

# Initialize the LLaMA model with optimized settings for limited RAM
def init_llama(model_path):
    return Llama(
        model_path=model_path,
        n_ctx=512,      # Reduce context to save memory (default is 2048)
        n_threads=4,    # Adjust based on your CPU (can even use 2)
        n_gpu_layers=0  # Use CPU only; increase this if you have GPU support
    )

# Generate response from LLaMA model
def get_llama_response(llama_model, prompt):
    output = llama_model(prompt, max_tokens=150, temperature=0.7, stop=["</s>"])
    return output["choices"][0]["text"].strip()
