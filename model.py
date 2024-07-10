import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv("crop recommendation _csv.csv")

# Select features and target
features = ['temperature', 'humidity', 'ph', 'rainfall']
X = data[features]
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model using K-nearest neighbors algorithm
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Save the model using pickle
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn, file)

# Get input from user
temperature = float(input("Enter the average temperature (in degrees Celsius): "))
humidity = float(input("Enter the average humidity (in percentage): "))
ph = float(input("Enter the soil pH: "))
rainfall = float(input("Enter the average rainfall (in millimeters): "))

# Load the model using pickle
with open('knn_model.pkl', 'rb') as file:
    knn = pickle.load(file)

# Predict crop based on user input
user_input = pd.DataFrame(np.array([[temperature, humidity, ph, rainfall]]), columns=features)
predicted_crop = knn.predict(user_input)

# Print recommended crop
print("Based on the information provided, the recommended crop is:", predicted_crop[0])
