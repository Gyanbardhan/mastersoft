from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('svc_model.pkl', 'rb'))
scaler_t = pickle.load(open('scaler_t.pkl', 'rb'))
scaler_mc = pickle.load(open('scaler_mc.pkl', 'rb'))
scaler_tc = pickle.load(open('scaler_tc.pkl', 'rb'))

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mapping dropdown values to integers
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        tech_support_map = {'No': 0, 'Yes': 1, 'No internet service': 2}
        payment_method_map = {
            'Electronic check': 0,
            'Mailed check': 1,
            'Bank transfer (automatic)': 2,
            'Credit card (automatic)': 3
        }
        online_security_map = {'No': 0, 'Yes': 1, 'No internet service': 2}
        paperless_billing_map = {'Yes': 0, 'No': 1}
        senior_citizen_map = {'No': 0, 'Yes': 1}

        # Get data from the form and convert the categorical values
        contract = contract_map[request.form['Contract']]

        mean_mc = scaler_mc.mean_[0]
        std_dev_mc = scaler_mc.scale_[0]
        monthly_charges = (float(request.form['MonthlyCharges']) - mean_mc) / std_dev_mc
        
        tech_support = tech_support_map[request.form['TechSupport']]

        mean_tc = scaler_tc.mean_[0]
        std_dev_tc = scaler_tc.scale_[0]
        total_charges = (float(request.form['TotalCharges']) - mean_tc) / std_dev_tc

        mean_t = scaler_t.mean_[0]
        std_dev_t = scaler_t.scale_[0]
        tenure = (int(request.form['tenure']) - mean_t) / std_dev_t

        avg_monthly_charge = total_charges / tenure
        payment_method = payment_method_map[request.form['PaymentMethod']]
        online_security = online_security_map[request.form['OnlineSecurity']]
        senior_citizen = senior_citizen_map[request.form['SeniorCitizen']]
        paperless_billing = paperless_billing_map[request.form['PaperlessBilling']]

        # Combine all the processed inputs into a list
        input_data = [
            contract,
            monthly_charges,
            tech_support,
            avg_monthly_charge,
            total_charges,
            tenure,
            payment_method,
            online_security,
            senior_citizen,
            paperless_billing
        ]

        # Convert to a numpy array and reshape for prediction
        features_array = np.array(input_data).reshape(1, -1)
        
        # Make prediction using the loaded model
        prediction = model.predict(features_array)
        
        # Return the prediction result
        output = 'Churn' if prediction[0] == 1 else 'Not Churn'
        
        return render_template('index.html', prediction_text=f'Customer will: {output}', **request.form)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
