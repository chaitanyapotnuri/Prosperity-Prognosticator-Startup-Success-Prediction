from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('random_forest_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get the values the user actually typed in
    user_inputs = [float(x) for x in request.form.values()]
    
    # 2. Fill the rest of the 33 features with 0.0 (Padding)
    # This prevents the "Size Mismatch" error
    total_features_needed = 33
    final_features = user_inputs + [0.0] * (total_features_needed - len(user_inputs))
    
    # 3. Convert to numpy array and reshape for the model
    prediction_data = np.array(final_features).reshape(1, -1)
    
    # 4. Make prediction
    prediction = model.predict(prediction_data)
    
    output = "SUCCESS: Likely to be Acquired" if prediction[0] == 1 else "RISK: Likely to Close"
    
    return render_template('result.html', prediction_text=output)
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)