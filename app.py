from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model (ensure model.pkl exists in the same directory)
model = joblib.load('model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Home Route (Simple check to confirm the server is running)
@app.route('/')
def index():
    return "Vehicle Service Prediction API is running!"

# Prediction Route (where the machine learning model will be used)
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the request (data sent as JSON)
    data = request.get_json()

    # Ensure the incoming data has the necessary fields
    try:
        km_driven = data['km_driven']
        days_since_last_service = data['days_since_last_service']
        vehicle_type = data['vehicle_type']  # 0 for car, 1 for bike
        service_type = data['service_type']  # 0: repair, 1: wash, 2: full_service
    except KeyError:
        return jsonify({"error": "Missing required parameters"}), 400

    # Prepare data for prediction
    input_data = np.array([[km_driven, days_since_last_service, vehicle_type, service_type]])

    # Use the model to predict the next service due in days
    prediction = model.predict(input_data)

    # Return the prediction as a JSON response
    return jsonify({'predicted_days': int(prediction[0])})

# Run the Flask app on localhost
if __name__ == '__main__':
    app.run(debug=True)
