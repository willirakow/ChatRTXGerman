# ChatRTXGerman
## Chat with RTX based on a German LLM

NVIDIA Chat with RTX is a popular tool that demonstrates the capabilities of chatbots based on an LLM-RAG setup.

It requires a Windows PC with an RTX GPU with at least 8GB of VRAM.
[NVIDIA chat with RTX](https://www.nvidia.com/en-us/ai-on-rtx/chatrtx/)

This project demonstrates the use of an LLM supporting German.The demo uses [LAION LeoLM](https://huggingface.co/LeoLM)

## Setup

### Install Chat with RTX.
Check the system requirements and download [here](https://www.nvidia.com/en-us/ai-on-rtx/chatrtx/).

Check that everything is working by running the tool.

### Identify Installation directory
The default directory is C:\Users\username\AppData\Local\NVIDIA\ChatWithRTX

Replace username in the following commands by the name used on your PC.

### Clone the German model leo-hessianai-7b-chat

Install [git for Windows](https://gitforwindows.org/) and run Git Bash
```
cd /c/Users/username/AppData/Local/NVIDIA/ChatWithRTX/RAG/trt-llm-rag-windows-main/model
git clone https://huggingface.co/LeoLM/leo-hessianai-7b-chat
```

### Convert the model with TensorRT-LLM

Open the Anaconda Prompt
```
cd C:\Users\username\AppData\Local\NVIDIA\ChatWithRTX
```

Check available environments and activate the TensorRT-LLM environment
```
conda info --envs
conda activate  C:\Users\username\AppData\Local\NVIDIA\ChatWithRTX\env_nvd_rag
```

Convert the model
```
python ./TensorRT-LLM/TensorRT-LLM-0.7.0/examples/llama/build.py --model_dir ./RAG\trt-llm-rag-windows-main\model\leo-hessianai-7b-chat --dtype float16 --remove_input_padding --use_gpt_attention_plugin float16 --enable_context_fmha --use_gemm_plugin float16 --output_dir ./RAG\trt-llm-rag-windows-main\model\leo-hessianai-7b-chat
```

### Modify the start-up configuration of Chat with RTX by adding the links to LeoLM
```
C:\Users\username\AppData\Local\NVIDIA\ChatWithRTX\RAG\trt-llm-rag-windows-main\config
```
Add to config.json:

            {
                "name": "LeoLM",
                "installed": true,
                "metadata": {
                    "model_path": "model\\leo-hessianai-7b-chat",
                    "engine": "llama_float16_tp1_rank0.engine",
                    "tokenizer_path": "model\\leo-hessianai-7b-chat",
                    "max_new_tokens": 512,
                    "max_input_token": 2048,
                    "temperature": 0.1
                }
            }
