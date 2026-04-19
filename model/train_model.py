import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

print("Model training...")

X = []
y = []
emotions = ['happy','sad','angry','neutral','fearful']

for emotion in emotions:
    for i in range(300):
        features = np.random.randn(100)
        X.append(features)
        y.append(emotion)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# CREATE FOLDER + SAVE
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/emotion_model.pkl')
print("Model saved!")