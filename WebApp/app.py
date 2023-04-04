from flask import Flask, render_template, request, jsonify
from helper import preprocess, LassoRegressionModel
import json

app = Flask(__name__)

# routes
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods = ["POST"])
def predict():
    data = request.form.get("json_model_inputdata")

    preprocessed = preprocess(data)

    predicted_price = LassoRegressionModel.predict(preprocessed)

    return render_template("predict.html", predicted_price=predicted_price)
if __name__ == "__main":
    app.run(debug=True)