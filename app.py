from flask import Flask, request, render_template, url_for
import pickle
import numpy as np
import xgboost as xgb

app = Flask(__name__, static_folder="static")

# Load the trained model and scaler
model = pickle.load(open("models/xgboost_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Route for introduction page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for prediction input page
@app.route('/predict_page', methods=['GET'])
def predict_page():
    return render_template('home.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        final_input = scaler.transform(np.array(data).reshape(1, -1))
        prediction = model.predict(final_input)[0]
        result = "Fake" if prediction == 1 else "Real"
        return render_template('home.html', prediction_text=f"The profile is {result}")
    except Exception as e:
        return render_template('home.html', prediction_text="Error: Invalid input data")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
