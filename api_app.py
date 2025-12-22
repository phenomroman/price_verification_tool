import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from process import process_model

app = FastAPI(title='Price Verification API')

class data_input(BaseModel):
  input_list: list = []
  code: str = None
  tolerance: float = 0.15

class output(BaseModel):
  result: dict

@app.post('/verify', response_model=output)
async def verify_price(data: data_input):
  result = process_model(data.input_list, data.code, data.tolerance)
  return output(result=result)

if __name__ == '__main__':
  uvicorn.run(app, host='0.0.0.0', port=8000)
