from fastapi import HTTPException, Depends
from fastapi.security import APIKeyHeader
import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor
import os

# API keys with their types
API_KEYS = {
    "api_key_123r": {"type": "testing"},
    "api_key_123y": {"type": "testing"},
    "api_key_127r": {"type": "production"},
    "api_key_127m": {"type": "staging"}
}

api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return API_KEYS[api_key]['type']

model = None
processor = None

def init_model():
    global model, processor
    model_path = os.getenv("MODEL_PATH", "/path/to/model")  # Default path is set if ENV is not set
    model = AutoModelForImageClassification.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    processor = ViTImageProcessor.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)