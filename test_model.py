from sklearn.datasets import make_classification
from ml.model import train_model

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

model = train_model(X, y)

print(F"Model trained: {type(model)}")