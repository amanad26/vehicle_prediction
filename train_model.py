import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Sample dataset
# data = {
#     'km_driven': [10000, 20000, 15000, 18000 ,16000 , 17000, 19000, 14000, 13000],
#     'days_since_last_service': [60, 90, 45, 80 , 10, 20, 40, 100, 36],
#     'vehicle_type': ['car', 'car', 'bike', 'car','car', 'car', 'bike', 'car' , 'car'],
#     'service_type': ['repair', 'wash', 'repair', 'full_service','repair', 'wash', 'repair', 'full_service','full_service'],
#     'next_service_due_days': [30, 20, 60, 15, 40, 60,50,10,90]
# }
data = pd.read_csv('vehicle_service_data.csv')
df = pd.DataFrame(data)

# Encode categorical columns
df['vehicle_type'] = df['vehicle_type'].map({'car': 0, 'bike': 1})
df['service_type'] = df['service_type'].map({'repair': 0, 'wash': 1, 'full_service': 2})

X = df[['km_driven', 'days_since_last_service', 'vehicle_type', 'service_type']]
y = df['next_service_due_days']

model = RandomForestRegressor().fit(X, y)

joblib.dump(model, 'model.pkl')  # Save model
