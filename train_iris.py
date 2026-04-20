import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(4,)),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(device_name, X_train, y_train, X_test, y_test, epochs=50):
    with tf.device(device_name):
        model = build_model()
        print(f"\n{'='*50}")
        print(f"Training on: {device_name}")
        print(f"{'='*50}")
        
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=8,
            validation_split=0.2,
            verbose=0
        )
        end_time = time.time()
        
        train_time = end_time - start_time
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Training time: {train_time:.4f} seconds")
        print(f"Final accuracy: {test_accuracy:.4f}")
        print(f"Final loss: {test_loss:.4f}")
        
        return {
            'device': device_name,
            'time': train_time,
            'accuracy': test_accuracy,
            'loss': test_loss
        }

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("="*50)
print("CPU vs GPU Training Comparison")
print("="*50)
print(f"Dataset: Iris ({len(X)} samples, 4 features)")
print(f"Epochs: 50")

cpu_result = train_model('/CPU:0', X_train, y_train, X_test, y_test)

gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
if gpu_available:
    gpu_result = train_model('/GPU:0', X_train, y_train, X_test, y_test)
    
    print(f"\n{'='*50}")
    print("COMPARISON RESULTS")
    print(f"{'='*50}")
    print(f"CPU Time:  {cpu_result['time']:.4f}s")
    print(f"GPU Time:  {gpu_result['time']:.4f}s")
    speedup = cpu_result['time'] / gpu_result['time'] if gpu_result['time'] > 0 else 0
    print(f"Speedup:   {speedup:.2f}x")
    print(f"\nCPU Accuracy: {cpu_result['accuracy']:.4f}")
    print(f"GPU Accuracy: {gpu_result['accuracy']:.4f}")
else:
    print("\n[No GPU detected - CPU training only]")
