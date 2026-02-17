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
    user_inputs = [float(x) for x in request.form.values()]
    
    total_features_needed = 33
    final_features = user_inputs + [0.0] * (total_features_needed - len(user_inputs))
    prediction_data = np.array(final_features).reshape(1, -1)
    probabilities = model.predict_proba(prediction_data)
    prediction = 1 if probabilities[0][1] > 0.3 else 0 
    output = "SUCCESS: Likely to be Acquired" if prediction== 1 else "RISK: Likely to Close" 
    return render_template('result.html', prediction_text=output)
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)