from huggingface_hub import HfFolder
import os
import torch
from transformers import pipeline

# Đặt token của bạn vào đây
hf_token = os.getenv("HF_TOKEN", "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")
# Lưu token vào local
HfFolder.save_token(hf_token)

from huggingface_hub import login

hf_access_token = "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"
login(token=hf_access_token)

if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    print("CUDA is available.")

    _model = pipeline(
        "text-generation",
        model="tonyshark/llama-3.2-nemotron-3b-instruct",
        torch_dtype=dtype,
        device_map="auto",  # Hoặc có thể thử "cpu" nếu không ổn,
        max_new_tokens=256,
        token="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"
    )
else:
    print("No GPU available, using CPU.")
    _model = pipeline(
        "text-generation",
        model="tonyshark/llama-3.2-nemotron-3b-instruct",
        device_map="cpu",
        max_new_tokens=256,
        token="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"
    )
