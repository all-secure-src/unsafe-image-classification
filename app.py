from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from PIL import Image
import io
import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor
import os
import base64

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

# Check for GPU availability and set up DataParallel if multiple GPUs are available
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = os.getenv("MODEL_PATH", "unsafe-image-classification")
token = os.getenv("TOKEN", "")
print("Args: ", {"model_path": model_path, "token": token})

model = AutoModelForImageClassification.from_pretrained(model_path, torch_dtype=torch.float16, token=token)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)
processor = ViTImageProcessor.from_pretrained(model_path, torch_dtype=torch.float16, token=token)

app = FastAPI()

class ImageData(BaseModel):
    image_bytes: str

@app.post("/unsafe-image-classification/")
async def classify_image(image_data: ImageData, api_key_type: str = Depends(get_api_key)):
    try:
        image_bytes = base64.b64decode(image_data.image_bytes)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

        safe_score, unsafe_score = probabilities[0]
        final_result = "unsafe" if unsafe_score > safe_score else "safe"

        return {
            "label": final_result,
            "score": {"safe": float(safe_score), "unsafe": float(unsafe_score)}
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except IOError:
        raise HTTPException(status_code=400, detail="Invalid image data provided.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)