from flask import Flask, render_template, request
import pandas as pd
import joblib


app = Flask(__name__)


# Load saved files
model = joblib.load("model/cancer_model.pkl")
scaler = joblib.load("model/scaler.pkl")
selector = joblib.load("model/selector.pkl")
encoder = joblib.load("model/label_encoder.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None


    if request.method == "POST":

        file = request.files["file"]


        if file:

            # Read uploaded CSV
            data = pd.read_csv(file)

            # Convert to array
            values = data.values

            # Apply preprocessing
            values = scaler.transform(values)

            values = selector.transform(values)

            # Predict class
            pred = model.predict(values)

            prediction = encoder.inverse_transform(pred)[0]

            # Predict probability
            probabilities = model.predict_proba(values)

            confidence = round(
                max(probabilities[0]) * 100,
                2
            )


    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)