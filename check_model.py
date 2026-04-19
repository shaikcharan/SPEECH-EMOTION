import pickle

with open("speech_emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

print(type(model))
print(model)