import joblib

model = joblib.load("models/model.pkl")

vectorizer = joblib.load("models/vectorizer.pkl")


def predict_sentiment(text):

    text = vectorizer.transform([text])

    prediction = model.predict(text)

    if prediction[0] == 1:
        return "Positive 😊"

    return "Negative 😔"