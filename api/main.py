import sys
import os

# Ensure the parent directory is in the path to import from core
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import io
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Body, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union
from core.models import inference_engine
from core.constants import ALL_FEATURES
from api.security import get_api_key

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
    api_key: str = Depends(get_api_key),
    data: DataInput = Body(
        ...,
        openapi_examples={
            "dictionary_input": {
                "summary": "Dictionary Input (Recommended)",
                "description": "Send a key-value pair dictionary where keys match feature names.",
                "value": {
                    "input_data": {
                        "YEAR": 2024,
                        "QUANTITY": 100.0,
                        "TENOR OF PAYMENT": 30,
                        "FREIGHT CHARGES": 500.0,
                        "EXPORTER": "Global Trade Inc",
                        "EXPORTER'S COUNTRY": "CHINA, PEOPLE’S REPUBLIC OF",
                        "IMPORTER": "Local Goods Ltd",
                        "COUNTRY_OF_ORIGIN": "CHINA, PEOPLE’S REPUBLIC OF",
                        "CURRENCY": "USD",
                        "TRADE-TERM": "FOB",
                        "SHIPMENT FROM": "SHANGHAI",
                        "SHIPMENT TO": "CHITTAGONG"
                    },
                    "code": "52094200",
                    "tolerance": 0.15
                }
            },
            "list_input": {
                "summary": "List Input (Legacy)",
                "description": "Send a list of values in the specific feature order.",
                "value": {
                    "input_data": [2024, 100.0, 30, 500.0, "Global Trade Inc", "CHINA, PEOPLE’S REPUBLIC OF", "Local Goods Ltd", "CHINA, PEOPLE’S REPUBLIC OF", "USD", "FOB", "SHANGHAI", "CHITTAGONG"],
                    "code": "52094200",
                    "tolerance": 0.15
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

@app.post('/predict/excel')
async def predict_excel(
    file: UploadFile = File(...), 
    tolerance: float = 0.15,
    api_key: str = Depends(get_api_key)
):
    """
    Upload an Excel file to get batch predictions.
    The API returns an Excel file with added prediction columns.
    If the Excel file contains a 'tolerance' column, those values will be used for each row.
    Otherwise, the provided default tolerance (0.15) will be used.
    """
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an Excel file.")
    
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # Core model handling
        df_processed = inference_engine.predict_batch(df, goods_code_col='HSCODE', tolerance=tolerance)
        
        # Create output buffer
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_processed.to_excel(writer, index=False)
        
        output.seek(0)
        
        filename = f"predictions_{file.filename}"
        headers = {
            'Content-Disposition': f'attachment; filename="{filename}"'
        }
        
        return StreamingResponse(
            output,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers=headers
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
