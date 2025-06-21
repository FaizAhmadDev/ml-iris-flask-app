import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("iris_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]
    prediction = model.predict([features])[0]
    iris_classes = ['Setosa', 'Versicolor', 'Virginica']
    return jsonify({"prediction": iris_classes[prediction]})

if __name__ == "__main__":
    app.run(debug=True)
