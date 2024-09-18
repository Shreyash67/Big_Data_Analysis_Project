from flask import Flask, render_template, request
import pickle
import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


app = Flask(__name__)


uri = "mongodb+srv://waralkarshayu:shreyash12345@cluster0.flbzs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client["mydatabase"]
collection = db["features"]

# Load the model
model_path = 'E:\\BDA_Project2\\best_model\\best_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define mappings for categorical features
cut_mapping = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
color_mapping = {'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7}
clarity_mapping = {'IF': 1, 'VVS1': 2, 'VVS2': 3, 'VS1': 4, 'VS2': 5, 'SI1': 6, 'SI2': 7, 'I1': 8}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and preprocess data from form
        features = {
            'carat': float(request.form.get('carat')),
            'depth': float(request.form.get('depth')),
            'table': float(request.form.get('table')),
            'x': float(request.form.get('x')),
            'y': float(request.form.get('y')),
            'z': float(request.form.get('z')),
            'cut': cut_mapping[request.form.get('cut')],
            'color': color_mapping[request.form.get('color')],
            'clarity': clarity_mapping[request.form.get('clarity')]
        }
        
        # Convert features to a list of lists (assuming the model expects this format)
        feature_values = [[features['carat'], features['depth'], features['table'],
                           features['x'], features['y'], features['z'],
                           features['cut'], features['color'], features['clarity']]]
        
        # Predict the price
        prediction = model.predict(feature_values)
        
        # Convert prediction to scalar if it's an array
        if isinstance(prediction, np.ndarray):
            prediction = prediction[0]
        
        # Ensure the prediction is a float for formatting
        if not isinstance(prediction, float):
            prediction = float(prediction)
        
        # Log the prediction result
        print(f"Prediction value: {prediction}")
        
        # Prepare data for MongoDB insertion
        diamond_data = {
            'Carat': features['carat'],
            'Depth': features['depth'],
            'Table': features['table'],
            'X': features['x'],
            'Y': features['y'],
            'Z': features['z'],
            'Cut': features['cut'],
            'Color': features['color'],
            'Clarity': features['clarity'],
            'Price': prediction  # Add the predicted price to the MongoDB document
        }
        
        # Insert features and the predicted price into MongoDB
        collection.insert_one(diamond_data)
        
        # Return formatted result
        return render_template('index.html', prediction_text=f'Predicted Diamond Price: ${prediction:,.2f}')
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text="An error occurred during prediction.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)