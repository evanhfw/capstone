import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

SEPAL_LENGTH = 5.1
SEPAL_WIDTH = 3.5
PETAL_LENGTH = 1.4
PETAL_WIDTH = 0.2

def main():
    model = tf.keras.models.load_model("iris_model.keras")
    iris = load_iris()
    
    scaler = StandardScaler()
    scaler.fit(iris.data)
    
    input_data = np.array([[SEPAL_LENGTH, SEPAL_WIDTH, PETAL_LENGTH, PETAL_WIDTH]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled, verbose=0)
    
    predicted_class_idx = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][predicted_class_idx])
    
    print(f"Input: sepal_length={SEPAL_LENGTH}, sepal_width={SEPAL_WIDTH}, petal_length={PETAL_LENGTH}, petal_width={PETAL_WIDTH}")
    print(f"\nPredicted class: {iris.target_names[predicted_class_idx]}")
    print(f"Confidence: {confidence:.4f}")
    print("\nProbabilities:")
    for i, class_name in enumerate(iris.target_names):
        print(f"  {class_name}: {prediction[0][i]:.4f}")

if __name__ == "__main__":
    main()
