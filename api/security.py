import os
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    """
    Validate the API key from the header against a list of allowed keys.
    The allowed keys should be provided in the API_KEYS environment variable
     as a comma-separated string.
    """
    api_keys_str = os.getenv("API_KEYS")
    
    if not api_keys_str:
        # Strict mode: If no keys are configured, deny all access
        raise HTTPException(
            status_code=500, 
            detail="API Security is not configured. Please set API_KEYS environment variable."
        )
    
    allowed_keys = [key.strip() for key in api_keys_str.split(",") if key.strip()]
    
    if api_key in allowed_keys:
        return api_key
    
    raise HTTPException(status_code=403, detail="Invalid API Key")
