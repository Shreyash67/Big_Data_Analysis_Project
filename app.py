from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

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
        
        # Predict
        prediction = model.predict(feature_values)
        
        # Convert prediction to scalar if it's an array
        if isinstance(prediction, np.ndarray):
            prediction = prediction[0]
        
        # Ensure the prediction is a float for formatting
        if not isinstance(prediction, float):
            prediction = float(prediction)
        
        # Return formatted result
        return render_template('index.html', prediction_text=f'Predicted Diamond Price: ${prediction:,.2f}')
    
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
