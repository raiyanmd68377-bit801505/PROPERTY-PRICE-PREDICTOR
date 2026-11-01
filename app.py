
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained ML model
with open("best_house_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "area": float(request.form["area"]),
        "bedrooms": int(request.form["bedrooms"]),
        "bathrooms": int(request.form["bathrooms"]),
        "stories": int(request.form["stories"]),
        "parking": int(request.form["parking"]),

        "mainroad_no": 0,
        "mainroad_yes": 1,
        "guestroom_no": 1,
        "guestroom_yes": 0,
        "basement_no": 1,
        "basement_yes": 0,
        "hotwaterheating_no": 1,
        "hotwaterheating_yes": 0,
        "airconditioning_no": 1,
        "airconditioning_yes": 0,
        "prefarea_no": 1,
        "prefarea_yes": 0,
        "furnishingstatus_furnished": 0,
        "furnishingstatus_semi-furnished": 0,
        "furnishingstatus_unfurnished": 1
    }

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
