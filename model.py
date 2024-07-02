import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load the csv file
Data = pd.read_csv("Crimes_reported_from_2011_to_2022.csv")

le_province = LabelEncoder()
le_district = LabelEncoder()
le_crime_type = LabelEncoder()

Data['Province'] = le_province.fit_transform(Data['Province'])
Data['District'] = le_district.fit_transform(Data['District'])
Data['Crime Type'] = le_crime_type.fit_transform(Data['Crime Type'])

Scaler = StandardScaler()
Scaler.fit(Data[['Year']])

# Create new feature
Data['Total Crimes'] = Data[['Crime Type']].sum(axis=1)

print(Data)

# Crime prediction for districts of kpk
kpk_data = Data[Data['Province'] == 0]

X_kpk = Data[['Year', 'Province', 'District', 'Crime Type']]
y_kpk = Data['Total Crimes']
X_train_kpk, X_test_kpk, y_train_kpk, y_test_kpk = train_test_split(X_kpk, y_kpk, test_size=0.2, random_state=42)

model_kpk = RandomForestRegressor()
model_kpk.fit(X_train_kpk, y_train_kpk)

# Predict for future years (2023-2030)
future_years = range(2023, 2031)  # 2023 to 2030 inclusive
crime_type = {}

future_predictions = pd.DataFrame(columns=['Year', 'Province', 'District', 'Crime Type', 'Predicted Crimes'])

for year in future_years:
    for district in X_kpk['District'].unique():
        for crime_code, crime_type in crime_type.items():
            future_data = pd.DataFrame({
                'Year': [year],
                'Province': [0],
                'District': [district],
                'Crime Type': [crime_code]
                })
            
    predicted_crimes = model_kpk.predict(future_data[['Year','Province', 'District', 'Crime Type']])
            
    future_predictions = future_predictions._append({
                'Year': year,
                'Province': 0,
                'District': district,
                'Crime Type': crime_type,
                'Predicted Crimes': predicted_crimes[0]  # Assuming single prediction per entry
            }, ignore_index=True)

# Print or save future predictions DataFrame
print(future_predictions)

# Save the trained model using joblib
joblib.dump(model_kpk, 'model_kpk.pkl')

print("Model trained and future predictions completed.")



# Crime prediction for districts of punjab
punjab_data = Data[Data['Province'] == 1]

X_punjab = Data[['Year', 'Province', 'District', 'Crime Type']]
y_punjab = Data['Total Crimes']
X_train_punjab, X_test_punjab, y_train_punjab, y_test_punjab = train_test_split(X_punjab, y_punjab, test_size=0.2, random_state=42)

model_punjab = RandomForestRegressor()
model_punjab.fit(X_train_punjab, y_train_punjab)

# Predict for future years (2023-2030)
future_years = range(2023, 2031)  # 2023 to 2030 inclusive

future_predictions = pd.DataFrame(columns=['Year', 'Province', 'District', 'Crime Type', 'Predicted Crimes'])

for year in future_years:
    for district in X_punjab['District'].unique():
        for crime_code, crime_type in le_crime_type.items():
            future_data = pd.DataFrame({
                'Year': [year],
                'District': [district],
                'Crime Type': [crime_code],
                'Province': [1]  # Assuming Sindh province ID is 2
            })
            
            predicted_crimes = model_punjab.predict(future_data[['Year','Province', 'District', 'Crime Type']])
            
            future_predictions = future_predictions._append({
                'Year': year,
                'Province': 1,
                'District': district,
                'Crime Type': crime_type,
                'Predicted Crimes': predicted_crimes[0]  # Assuming single prediction per entry
            }, ignore_index=True)

# Print or save future predictions DataFrame
print(future_predictions)

# Save the trained model using joblib
joblib.dump(model_punjab, 'model_punjab.pkl')

print("Model trained and future predictions completed.")



# Crime prediction for districts of sindh
sindh_data = Data[Data['Province'] == 2]

X_sindh = Data[['Year', 'Province', 'District', 'Crime Type']]
y_sindh = Data['Total Crimes']
X_train_sindh, X_test_sindh, y_train_sindh, y_test_sindh = train_test_split(X_sindh, y_sindh, test_size=0.2, random_state=42)

model_sindh = RandomForestRegressor()
model_sindh.fit(X_train_sindh, y_train_sindh)

# Predict for future years (2023-2030)
future_years = range(2023, 2031)  # 2023 to 2030 inclusive


future_predictions = pd.DataFrame(columns=['Year', 'Province', 'District', 'Crime Type', 'Predicted Crimes'])

for year in future_years:
    for district in X_sindh['District'].unique():
        for crime_code, crime_type in le_crime_type.items():
            future_data = pd.DataFrame({
                'Year': [year],
                'District': [district],
                'Crime Type': [crime_code],
                'Province': [2]  # Assuming Sindh province ID is 2
            })
            
            predicted_crimes = model_sindh.predict(future_data[['Year','Province', 'District', 'Crime Type']])
            
            future_predictions = future_predictions.dropna()
            future_predictions = future_predictions._append({
                'Year': year,
                'Province': 2,
                'District': district,
                'Crime Type': crime_type,
                'Predicted Crimes': predicted_crimes[0]  # Assuming single prediction per entry
            }, ignore_index=True)

# Print or save future predictions DataFrame
print(future_predictions)

# Save the trained model using joblib
joblib.dump(model_sindh, 'model_sindh.pkl')

print("Model trained and future predictions completed.")
