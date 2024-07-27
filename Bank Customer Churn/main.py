from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Create FastAPI instance
app = FastAPI()

# Load models
lr_model = joblib.load("models/LR.sav")
rf_model = joblib.load("models/random_forest.sav")
svm_model = joblib.load("models/SVM_model.sav")

# Define a Pydantic model for the input data
class InputData(BaseModel):
    credit_score: int
    country: str
    gender: str
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: bool
    active_member: bool
    estimated_salary: float
    model_type: str

# Define prediction endpoint
@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Extract features from input data
        input_data = [
            data.credit_score, data.country, data.gender, data.age,
            data.tenure, data.balance, data.products_number, data.credit_card,
            data.active_member, data.estimated_salary
        ]

        # Select model based on input
        if data.model_type == "Logistic Regression":
            model = lr_model
        elif data.model_type == "Random Forest":
            model = rf_model
        elif data.model_type == "SVM":
            model = svm_model
        else:
            raise HTTPException(status_code=400, detail="Model type not supported")

        prediction = model.predict([input_data])
        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
