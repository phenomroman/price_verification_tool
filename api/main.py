import sys
import os

# Ensure the parent directory is in the path to import from core
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from core.models import inference_engine
from core.constants import ALL_FEATURES

app = FastAPI(title='Price Verification API')

class DataInput(BaseModel):
    input_list: List[Any]  # List of values corresponding to ALL_FEATURES
    code: str
    tolerance: float = 0.15 # Kept for backward compatibility, though logic is currently hardcoded in models.py

class Output(BaseModel):
    result: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "Price Verification API is running"}

@app.post('/predict', response_model=Output)
async def predict_price(data: DataInput):
    if len(data.input_list) != len(ALL_FEATURES):
        raise HTTPException(status_code=400, detail=f"Input list must have {len(ALL_FEATURES)} elements. Got {len(data.input_list)}.")
    
    # Map list to dict
    input_data = dict(zip(ALL_FEATURES, data.input_list))
    
    result = inference_engine.predict(input_data, data.code, tolerance=data.tolerance)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return Output(result=result)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
