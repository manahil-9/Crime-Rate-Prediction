from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
Data = pd.read_csv("Crimes_reported_from_2011_to_2022.csv")

app = Flask(__name__)

# Load models
try:
    model_kpk = joblib.load('model_kpk.pkl')
    model_punjab = joblib.load('model_punjab.pkl')
    model_sindh = joblib.load('model_sindh.pkl')
except Exception as e:
    print(f"Error loading models: {e}")
    model_kpk = model_punjab = model_sindh = None

# Initialize label encoders
province_encoder = LabelEncoder()
district_encoder = LabelEncoder()
crime_type_encoder = LabelEncoder()

# Fit encoders on data
province_encoder.fit(Data['Province'])
district_encoder.fit(Data['District'])
crime_type_encoder.fit(Data['Crime Type'])

@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    # Retrieve input data from the form
    province = request.form.get('Province')
    district = request.form.get('District')
    crime_type = request.form.get('Crime Type')
    year = request.form.get('Year')

    # Debug: Print the retrieved form data
    print(f"Province: {province}")
    print(f"District: {district}")
    print(f"Crime Type: {crime_type}")
    print(f"Year: {year}")

    # Validate inputs
    if not province:
        print("Error: Province is missing")
        return "Province is required", 400
    if not district:
        print("Error: District is missing")
        return "District is required", 400
    if not crime_type:
        print("Error: Crime Type is missing")
        return "Crime Type is required", 400
    if not year:
        print("Error: Year is missing")
        return "Year is required", 400

    # Select appropriate model based on the province
    model = None
    if province == 'KPK':
        model = model_kpk
    elif province == 'Punjab':
        model = model_punjab
    elif province == 'Sindh':
        model = model_sindh

    if model is None:
        print("Error: Province not recognized or model not available")
        return "Province not recognized or model not available", 400

    try:
        # Encode input data with default handling for unseen values
        encoded_province = province_encoder.transform([province])[0] if province in province_encoder.classes_ else -1
        encoded_district = district_encoder.transform([district])[0] if district in district_encoder.classes_ else -1
        encoded_crime_type = crime_type_encoder.transform([crime_type])[0] if crime_type in crime_type_encoder.classes_ else -1

        # Check if any encoded value is -1 indicating an unknown category
        if encoded_province == -1 or encoded_district == -1 or encoded_crime_type == -1:
            print("Error: Unrecognized category in input")
            return "Error: Unrecognized category in input", 400

        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'Year': [int(year)],
            'Province': [encoded_province],
            'District': [encoded_district],
            'Crime Type': [encoded_crime_type]
        })

        # Perform prediction
        prediction = model.predict(input_data)[0]

        # Display the prediction result
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Error during prediction: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)