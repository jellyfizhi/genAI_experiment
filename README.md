# genAI_experiment
---
![Chat Example](images/chatINTRO.jpeg)
Do you also want your very own chatbot? 
## Preparation/Requirements
- Python 3.11.12 (or similar)
- T4 GPU (from Google Colab or any GPU RAM that is at least equivalent to 15GB)
- Granted access to llama-3-8b model

- [Access request to a model on Hugging Face](https://huggingface.co/docs/hub/en/models-gated)


- [Adding a model to your collection](https://huggingface.co/docs/hub/en/collections)
![Sign up or login with your account](images/grantAccess1.png)
![Read through community license agreement and fill in details](images/grantAccess2.png)

### Personal experience - just in case:
- torch==2.6.0 (torchvision==0.21.0+cu124)
- numpy==2.2.5 (note: if there is numpy version error, set numpy version to a lower version before upgrading version to latest after installing other packages - transformers, bitsandbytes, accelerate)
- jedi>=0.16 (I used 0.16)
- accelerate==1.6.0
- transformers==4.51.3
---
## LLM download
```from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
save_path="/llama3"
# add your hugging face token here in the form hf_token = "xxxxxxxxxx"
def download_LLM(model_name, save_path, hf_token):
    """
    Downloads a LLM model from Hugging Face and saves it locally.

    Parameters:
        model_name (str): The name of the llama-3-8b model to download. Options:
                          - 'llama3' (small)
                          - 'llama3-medium'
                          - 'llama3-large'
                          - 'llama3-xl'
        save_path (str): Directory to save the model and tokenizer.
    """
    snapshot_path = snapshot_download(
    repo_id=model_name,
    token=hf_token,
    local_dir=save_path
    )

# print(f"Private model snapshot downloaded to: {snapshot_path}")

    print(f"✅ LLM model ('{model_name}') downloaded and saved to '{save_path}'.")

# Example usage
#download_LLM(model_name, save_path, hf_token)
```
---
## Deploy LLM locally
```from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from accelerate import load_checkpoint_and_dispatch

# Clear GPU memory before loading
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Collect any lingering memory fragments
        print(f"GPU Memory Cleared. Free: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
# Install required libraries if not already installed:
# pip install bitsandbytes accelerate

# Load the model and tokenizer with quantization
def load_LLM(save_path):
    """
    Loads the model and tokenizer from a saved local path with quantization.
    """
    # Clear GPU memory before loading
    clear_gpu_memory()

    # Define 4-bit quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Enable 4-bit quantization (lower than 8-bit)

        bnb_4bit_quant_type="nf4",  # Use NF4 quantization (optimized for LLMs)
        bnb_4bit_use_double_quant=True,  # Double quantization for extra savings
        bnb_4bit_compute_dtype="float16",  # Compute in FP16
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_path)

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        save_path,
        quantization_config=quantization_config,  # Pass the 4-bit quantization config
        device_map="auto",  # Automatically split across GPU/CPU
        low_cpu_mem_usage=True,  # Minimize CPU memory spikes
    )

    # Optional: Offload to CPU/disk if GPU memory is still insufficient
 #   model = load_checkpoint_and_dispatch(
 #       model,
 #       checkpoint=save_path,
 #       device_map="auto",
 #       offload_folder="./offload",  # Directory for offloaded weights
 #   )

    return tokenizer, model

# Function to chat with LLM
def chat_with_LLM(save_path):
    # Load the model and tokenizer
    tokenizer, model = load_LLM(save_path)  # Adjust save_path as needed

    print("🤖 Chatbot: Hello! Type 'exit' to end the chat.")

    while True:
        user_input = input("🧑 You: ")
        if user_input.lower() == "exit":
            print("🤖 Chatbot: Goodbye!")
            break

        # Tokenize input and move to GPU if available
        input_ids = tokenizer.encode(user_input, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate a response
        output = model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # Optional: Adds variety to responses
            top_k=50,        # Optional: Improves response quality
            top_p=0.95       # Optional: Improves response quality
        )

        # Decode and print response
        response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"🤖 Chatbot: {response}")
```
So the first problem I encountered with deploying the LLM was that the model was using up too much RAM and the code was quickly using up all the system RAM and GPU RAM which meant that I was unable to progress and actually run the chatbot. Now this was a problem because I couldn't just simply upgrade to more RAM and upgrading to more RAM also meant that the general public could not follow along and run this code either, so I had to think of a solution. This is where the quantization came into play as quantization reduces the computational and memory costs of running the program by representing the weights and activations with low-precision data types like 4-bit integer instead of the usual 32-bit floating point which together paired with clearing GPU RAM before loading allowed me to successfully run the program and deploy the LLM locally.
---
## Running the chatbot
```# Run the chatbot
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
save_path="/llama3"
chat_with_LLM(save_path)
```

## The whole story
I loaded up a new Colab page and set runtime type to T4 GPU. Then I downloaded GPT2 but the chatbot was too outdated and mostly untrained which made it difficult to work with as it kept repeating phrases in each reply until a limit was reached. Therefore, I changed to a different model; Falcon, however although more up-to-date, the file size was too big to be downloaded onto Colab and despite changing the method of download to using the hugging face snapshot download method and quantisation method, the RAM was used up nonetheless before the download could be finished. Which brings us to the llama-v3-8b model which I downloaded from hugging face using a token and this model was not too great in file size and trained well enough to sustain a proper conversation.

## Future plans

## Notes to write up
falcon - too large, failed to download into colab (used up RAM) - used hugging face snapshot download method and quantisation method to load? (or was this for llama 3 8b?)
llama 3 8b - used hugging face token, downloaded and worked - trained
gpt2? - download worked and was able to chat with it but too outdated and mostly untrained - kept repeating phrases until limit for each reply reached
> why quantisation? (for package bitsandbytes) - ss without quantization to show error -> whats the solution? story tell
how to utilise the limited gpu ram (quantization and )
afterwards - can play around with top k temperature thing and other parameters to optimise


