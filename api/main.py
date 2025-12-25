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
from typing import List, Dict, Any, Optional, Union

from core.models import inference_engine
from core.constants import ALL_FEATURES

app = FastAPI(title='Price Verification API')

class DataInput(BaseModel):
    # Accept either a dictionary (direct) or a list (legacy/ordered)
    input_data: Union[Dict[str, Any], List[Any]]
    code: str
    tolerance: float = 0.15 

class Output(BaseModel):
    result: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "Price Verification API is running"}

@app.post('/predict', response_model=Output)
async def predict_price(data: DataInput):
    final_input_data = {}
    
    # Check type of input_data
    if isinstance(data.input_data, list):
        # Legacy mode: Map list to dict using ALL_FEATURES order
        if len(data.input_data) != len(ALL_FEATURES):
            raise HTTPException(status_code=400, detail=f"Input list must have {len(ALL_FEATURES)} elements. Got {len(data.input_data)}.")
        final_input_data = dict(zip(ALL_FEATURES, data.input_data))
    elif isinstance(data.input_data, dict):
        # Modern mode: Use dict directly
        final_input_data = data.input_data
    else:
        raise HTTPException(status_code=400, detail="Invalid input format. Must be a list or a dictionary.")
    
    result = inference_engine.predict(final_input_data, data.code, tolerance=data.tolerance)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return Output(result=result)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
