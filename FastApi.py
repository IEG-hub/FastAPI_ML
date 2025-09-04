# Code to generate FastAPI from ML project 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from typing import Dict, List, Optional
import os

# Initialize FastAPI app
app = FastAPI(
    title="Company Response Prediction API",
    description="API for predicting company responses in financial products",
    version="1.0.0"
)

# Pydantic models for request and response
class PredictionRequest(BaseModel):
    product: str = Field(..., description="Product type")
    issue: str = Field(..., description="Issue type")
    state: str = Field(..., description="US State code")
    company: str = Field(..., description="Company name")
    consumer_disputed: str = Field(..., description="Consumer disputed (Yes/No/Unknown)")
    timely_response: str = Field(..., description="Timely response (Yes/No)")
    days: int = Field(..., ge=0, le=64, description="Days to respond (0-64)")

class PredictionResponse(BaseModel):
    prediction_code: int = Field(..., description="Predicted response code")
    prediction_label: str = Field(..., description="Predicted response label")
    confidence: float = Field(..., description="Confidence percentage for main prediction")
    probabilities: Dict[str, float] = Field(..., description="All probabilities for each response type")

class HealthResponse(BaseModel):
    status: str
    message: str

# Global variables for model and encoders
model = 'C:/Users/TheBridge/Desktop/FastAPI_ML/models'
encoders = None

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model, encoders
    try:
        # Update this path to your actual model path
        path_model = 'C:/Users/TheBridge/Desktop/FastAPI_ML/models'
        file_name = 'xgb_final_model.pkl'
        model_path = os.path.join(path_model, file_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        encoders = None  # Not needed for this implementation
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None

# Encoding dictionaries
PROD_BRAND_ENCODED = {
    'Bank account or service': 0, 'Card': 1, 'Credit reporting': 2, 
    'Debt collection': 3, 'Loan': 4, 'Money transfers': 5, 'Mortgage': 6
}

ISSUE_ENCODED = {'Credit': 0, 'Debt': 1, 'Loan': 2, 'Others': 3}

STATE_ENCODED = {
    'DE': 1, 'PA': 2, 'NJ': 3, 'GA': 4, 'CT': 5, 'MA': 6, 'MD': 7, 'SC': 8,
    'NH': 9, 'VA': 10, 'NY': 11, 'NC': 12, 'RI': 13, 'VT': 14, 'KY': 15,
    'TN': 16, 'OH': 17, 'LA': 18, 'IN': 19, 'MS': 20, 'IL': 21, 'AL': 22,
    'ME': 23, 'MO': 24, 'AR': 25, 'MI': 26, 'FL': 27, 'TX': 28, 'IA': 29,
    'WI': 30, 'CA': 31, 'MN': 32, 'OR': 33, 'KS': 34, 'WV': 35, 'NV': 36,
    'NE': 37, 'CO': 38, 'ND': 39, 'SD': 40, 'MT': 41, 'WA': 42, 'ID': 43,
    'WY': 44, 'UT': 45, 'OK': 46, 'NM': 47, 'AZ': 48, 'AK': 49, 'HI': 50
}

COMPANY_ENCODED = {
    'Other': 9423, 'Equifax': 2021, 'Experian': 1919, 'TransUnion': 1505, 
    'Bank of America': 1481, 'Wells Fargo': 1410, 'JPMorgan Chase': 1239, 
    'Citibank': 994, 'Ocwen': 924, 'Nationstar Mortgage': 732, 'Capital One': 556, 
    'GE Capital Retail': 473, 'U.S. Bancorp': 419, 'Enhanced Recovery Company, LLC': 394, 
    'Green Tree Servicing, LLC': 393, 'Encore Capital Group': 387, 'Discover': 367, 
    'Navient': 357, 'Amex': 263, 'PNC Bank': 231, 'Portfolio Recovery Associates, Inc.': 228, 
    'TD Bank': 211, 'Select Portfolio Servicing, Inc': 210, 'Transworld Systems Inc.': 209, 
    'HSBC': 195, 'SunTrust Bank': 180, 'Ally Financial Inc.': 163, 
    'Santander Consumer USA': 130, 'Barclays': 113, 'Convergent Resources, Inc.': 113, 
    'Fifth Third Bank': 110, 'PayPal': 109, 'Seterus': 108, 'RBS Citizens': 105
}

CONSUMER_DISPUTED_ENCODED = {'Yes': 1, 'No': 0, 'Unknown': 2}
TIMELY_RESPONSE_ENCODED = {'Yes': 1, 'No': 0}

COMPANY_RESPONSE_DECODED = {
    0: 'Closed',
    1: 'Closed with explanation', 
    2: 'Closed with monetary relief',
    3: 'Closed with non-monetary relief', 
    4: 'In progress', 
    5: 'Untimely response'
}

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if model is not None:
        return HealthResponse(status="healthy", message="API is running and model is loaded")
    else:
        return HealthResponse(status="unhealthy", message="Model not loaded")

# Get available options endpoints
@app.get("/options/products")
async def get_products():
    """Get available product options"""
    return {"products": list(PROD_BRAND_ENCODED.keys())}

@app.get("/options/issues")
async def get_issues():
    """Get available issue options"""
    return {"issues": list(ISSUE_ENCODED.keys())}

@app.get("/options/states")
async def get_states():
    """Get available state options"""
    return {"states": list(STATE_ENCODED.keys())}

@app.get("/options/companies")
async def get_companies():
    """Get available company options"""
    return {"companies": list(COMPANY_ENCODED.keys())}

@app.get("/options/consumer-disputed")
async def get_consumer_disputed():
    """Get available consumer disputed options"""
    return {"consumer_disputed": list(CONSUMER_DISPUTED_ENCODED.keys())}

@app.get("/options/timely-response")
async def get_timely_response():
    """Get available timely response options"""
    return {"timely_response": list(TIMELY_RESPONSE_ENCODED.keys())}

# Main prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_company_response(request: PredictionRequest):
    """
    Predict company response based on input parameters
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Validate input values
        if request.product not in PROD_BRAND_ENCODED:
            raise HTTPException(status_code=400, detail=f"Invalid product: {request.product}")
        
        if request.issue not in ISSUE_ENCODED:
            raise HTTPException(status_code=400, detail=f"Invalid issue: {request.issue}")
        
        if request.state not in STATE_ENCODED:
            raise HTTPException(status_code=400, detail=f"Invalid state: {request.state}")
        
        if request.company not in COMPANY_ENCODED:
            raise HTTPException(status_code=400, detail=f"Invalid company: {request.company}")
        
        if request.consumer_disputed not in CONSUMER_DISPUTED_ENCODED:
            raise HTTPException(status_code=400, detail=f"Invalid consumer_disputed: {request.consumer_disputed}")
        
        if request.timely_response not in TIMELY_RESPONSE_ENCODED:
            raise HTTPException(status_code=400, detail=f"Invalid timely_response: {request.timely_response}")
        
        # Prepare input data
        input_data = pd.DataFrame({
            'Prod_brand_encoded': [PROD_BRAND_ENCODED[request.product]],
            'Issue_encoded': [ISSUE_ENCODED[request.issue]],
            'State_encoded': [STATE_ENCODED[request.state]],
            'Company_encoded': [COMPANY_ENCODED[request.company]],
            'Consumer_disputed_modify': [CONSUMER_DISPUTED_ENCODED[request.consumer_disputed]],
            'Timely_response_encoded': [TIMELY_RESPONSE_ENCODED[request.timely_response]],
            'Difference_Days': [request.days]
        })
        
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        # Prepare response
        prediction_code = int(prediction[0])
        prediction_label = COMPANY_RESPONSE_DECODED[prediction_code]
        confidence = float(prediction_proba[0][prediction_code] * 100)
        
        # Prepare probabilities dictionary
        probabilities = {}
        for i, prob in enumerate(prediction_proba[0]):
            response_type = COMPANY_RESPONSE_DECODED[i]
            probabilities[response_type] = round(float(prob * 100), 2)
        
        return PredictionResponse(
            prediction_code=prediction_code,
            prediction_label=prediction_label,
            confidence=round(confidence, 2),
            probabilities=probabilities
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Company Response Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "options": "/options/*",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)