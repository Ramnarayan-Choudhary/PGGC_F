import os
import torch

class Config:
    # Environment
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # Model
    # Default to a safe default if not provided, but ideally explicitly set
    MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct") 
    CACHE_DIR = os.getenv("CACHE_DIR", os.path.expanduser("~/.cache/huggingface"))
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Attack Parameters
    DEFAULT_ITERATIONS = 100
    DEFAULT_K_COSINE = 5
    
    @staticmethod
    def validate():
        if not Config.HF_TOKEN:
            print("Warning: HF_TOKEN environment variable is not set. Hugging Face models may fail to load.")
