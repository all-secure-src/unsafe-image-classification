from fastapi import FastAPI, HTTPException, Depends
from .dependencies import get_api_key, init_model, model, processor
from pydantic import BaseModel
from PIL import Image
import io
import torch

app = FastAPI()

class ImageData(BaseModel):
    image_bytes: bytes

@app.on_event("startup")
async def startup_event():
    init_model()

@app.post("/unsafe-image-classification/")
async def classify_image(image_data: ImageData, api_key_type: str = Depends(get_api_key)):
    try:
        if not isinstance(image_data.image_bytes, bytes):
            raise ValueError("image_bytes must be of type bytes.")

        image = Image.open(io.BytesIO(image_data.image_bytes))
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
        predicted_label_id = logits.argmax(-1).item()
        classification = "unsafe" if predicted_label_id == 1 else "safe"
        return {
            "status": 1,
            "message": "Success",
            "data": {
                "classification": classification
            },
            "code": 200
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except IOError:
        raise HTTPException(status_code=400, detail="Invalid image data provided.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))