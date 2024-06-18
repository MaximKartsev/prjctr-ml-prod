import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

model = joblib.load('model/trained_model.pkl')
app = FastAPI()

class Excerpt(BaseModel):
    text: str

@app.post("/predict")
def predict(excerpt: Excerpt):
    try:
        # Perform the prediction
        prediction = model.predict([excerpt.text])
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
