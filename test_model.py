"""
This file is used to test the functions in ml/model.py
"""
import sklearn
from sklearn.datasets import make_classification
from ml.model import train_model, save_model, load_model, inference, performance_on_categorical_slice
from ml.data import process_data
from sklearn import preprocessing
import os
import pandas as pd

"""
Run this command in the terminal to test the train_model() function:
    python test_model.py

The expected output is:
    Training the model...
    Model training complete.
    Model trained: <class 'sklearn.ensemble._forest.RandomForestClassifier'>
    Model training complete.
    Predictions: f[1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 0 1 0 1 1 1 1 0 1 1 1 0 1 1 1 1 1 0 1 1 0 0
    1 1 0 0 0 1 1 1 0 1 0 0 1 0 0 1 1 1 1 0 1 1 0 1 1 0 0 1 1 1 1 1 0 1 0 0 1
    0 1 1 1 0 1 1 1 0 0 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 1]
    Inference function test completed successfully!
    
"""

X_train, y_train = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
df = pd.DataFrame(X_train, columns=feature_names)
df["category"] = ["A" if i % 2 == 0 else "B" for i in range(len(df))]
df["target"] = y_train

model = train_model(X_train, y_train)

print(F"Model trained: {type(model)}")

model_path = "test_model.pkl"

save_model(model, model_path)
loaded_model = load_model(model_path)

assert isinstance(loaded_model, type(model)), "Loaded model type mismatch!"

X_test, _ = make_classification(n_samples=100, n_features=10, random_state=42)

predictions = inference(model, X_test)

print(f"Predictions: f{predictions}")

categorical_features = ['category']
label = "target"

x_processed, y_processed, encoder, lb = process_data(
    df, categorical_features=categorical_features, label=label, training=True
)

precision, recall, fbeta = performance_on_categorical_slice(
    df, "category", "A", categorical_features, label, encoder, lb, loaded_model
)

print(f"Performance on categorical slice (category=A): Precision={precision}, Recall={recall}, F-beta={fbeta}")
os.remove(model_path)



