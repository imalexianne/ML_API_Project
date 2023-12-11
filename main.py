##import libraries
from fastapi import FastAPI, Query
import uvicorn
import joblib
import pandas as pd

##create app object
app = FastAPI()

###create home
@app.get('/')
def home():
    return{'message':'Welcome to Sepsis Prediction Using Fastapi'}


##load the model
model = joblib.load("Model/best_rf_model.joblib")

# Endpoint to get predictions
@app.post("/predict")
def predict_sepsis(
    Plasma_glucose: int = Query(..., description="Plasma_glucose"),
    Blood_Work_R1: int = Query(..., description="Blood_Work_R1"),
    Blood_Pressure: int = Query(..., description="Blood_Pressure"),
    Blood_Work_R2: int = Query(..., description="Blood_Work_R2"),
    Blood_Work_R3: int = Query(..., description="Blood_Work_R3"),
    BMI: float = Query(..., description="BMI"),
    Blood_Work_R4: float = Query(..., description="Blood_Work_R4"),
    Age: int = Query(..., description="Age")
   
):
    try:
        # Convert input data to the format expected by the model
        input_data = pd.DataFrame([{
            "Plasma_glucose": Plasma_glucose,
            "Blood_Work_R1": Blood_Work_R1,
            "Blood_Pressure": Blood_Pressure,
            "Blood_Work_R2": Blood_Work_R2,
            "Blood_Work_R3": Blood_Work_R3,
            "BMI": BMI,
            "Blood_Work_R4": Blood_Work_R4,
            "Age": Age
        }])

        # Make prediction
        prediction = model.predict(input_data)[0]

        sepsis_status = "Patient has sepsis" if prediction == 1 else "Patient does not have sepsis"

        # Return the prediction
        return {"prediction": sepsis_status}
    
    except Exception:
        # Handle other exceptions during prediction
        error_message = f"An error occurred during prediction"
        return {"error": error_message}