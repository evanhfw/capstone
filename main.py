from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

app = FastAPI(title="Iris Classifier API")

model = tf.keras.models.load_model("iris_model.keras")
iris = load_iris()

scaler = StandardScaler()
scaler.fit(iris.data)

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict

@app.get("/")
def root():
    return {"message": "Iris Classifier API", "classes": list(iris.target_names)}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    try:
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled, verbose=0)
        
        predicted_class_idx = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_class_idx])
        
        probabilities = {
            iris.target_names[i]: float(prediction[0][i])
            for i in range(len(iris.target_names))
        }
        
        return PredictionResponse(
            predicted_class=iris.target_names[predicted_class_idx],
            confidence=confidence,
            probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}
