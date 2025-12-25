import sys
import os

# Ensure the parent directory is in the path to import from core
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

from core.models import inference_engine
from core.constants import ALL_FEATURES

app = FastAPI(title='Price Verification API')

class DataInput(BaseModel):
    # Accept either a dictionary (direct) or a list (legacy/ordered)
    input_data: Union[Dict[str, Any], List[Any]] = Field(
        ..., 
        description="Prediction input data as a key-value dictionary or an ordered list."
    )
    code: str = Field(..., description="The HS code for the goods.", example="52094200")
    tolerance: float = Field(0.15, description="Prediction tolerance.")

class Output(BaseModel):
    result: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "Price Verification API is running"}

@app.post('/predict', response_model=Output)
async def predict_price(
    data: DataInput = Body(
        ...,
        openapi_examples={
            "dictionary_input": {
                "summary": "Dictionary Input (Recommended)",
                "description": "Send a key-value pair dictionary where keys match feature names.",
                "value": {
                    "input_data": {
                        "YEAR": int,
                        "QUANTITY": float,
                        "TENOR OF PAYMENT": int,
                        "FREIGHT CHARGES": float,
                        "EXPORTER": str,
                        "EXPORTER'S COUNTRY": str,
                        "IMPORTER": str,
                        "COUNTRY_OF_ORIGIN": str,
                        "CURRENCY": str,
                        "TRADE-TERM": str,
                        "SHIPMENT FROM": str,
                        "SHIPMENT TO": str
                    },
                    "code": str,
                    "tolerance": float
                }
            },
            "list_input": {
                "summary": "List Input (Legacy)",
                "description": "Send a list of values in the specific feature order.",
                "value": {
                    "input_data": [int, float, int, float, str, str, str, str, str, str, str, str],
                    "code": str,
                    "tolerance": float
                }
            }
        }
    )
):
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
