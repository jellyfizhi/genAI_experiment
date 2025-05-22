# genAI_experiment
---
![Chat Example](images/chatINTRO.jpeg)
## Preparation/Requirements
- Python 3.11.12 (or similar)
- 

### Personal experience - just in case:
- torch==2.6.0 (torchvision==0.21.0+cu124)
- numpy==2.2.5 (note: if there is numpy version error, set numpy version to a lower version before upgrading version to latest after installing other packages - transformers, bitsandbytes, accelerate)
- jedi>=0.16 (I used 0.16)
- accelerate==1.6.0
- transformers==4.51.3
---
## LLM download


## After
I loaded up a new Colab page and set runtime type to T4 GPU. Then I downloaded GPT2 but the chatbot was too outdated and mostly untrained which made it difficult to work with as it kept repeating phrases in each reply until a limit was reached. Therefore, I changed to a different model; Falcon, however although more up-to-date, the file size was too big to be downloaded onto Colab and despite changing the method of download to using the hugging face snapshot download method and quantisation method, the RAM was used up nonetheless before the download could be finished. Which brings us to the llama-v3-8b model which I downloaded from hugging face using a token and this model was not too great in file size and trained well enough to sustain a proper conversation.

falcon - too large, failed to download into colab (used up RAM) - used hugging face snapshot download method and quantisation method to load? (or was this for llama v3?)
llama v3 8b - used hugging face token, downloaded and worked - trained
gpt2? - download worked and was able to chat with it but too outdated and mostly untrained - kept repeating phrases until limit for each reply reached
> why quantisation? (for package bitsandbytes)

