from flask import Flask, request, jsonify, render_template
import pickle
from preprocessing import normalize_corpus


app = Flask(__name__)

model = pickle.load(open("sentiment_lr.pkl", "rb"))

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    str_features = " ".join([str(word) for word in request.form.values()])
    str_input = normalize_corpus(str_features)
    vectorized_input = vectorizer.transform([str_input])
    prediction = model.predict(vectorized_input)

    return render_template('index.html',
                           review_placeholder="Your Review:",
                           review=str_features,
                           prediction_placeholder="Polarity Prediction:",
                           polarity=("negative" if (prediction == 0) else "positive")
                           )

if __name__ == "__main__":
    app.run(debug=True)
