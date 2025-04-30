import pickle
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier  

# Load dataset
df = pd.read_csv("allusers.csv")  # Use the same dataset

# Select features and labels
selected_features = ['favourites_count', 'followers_count', 'statuses_count', 'friends_count', 'listed_count']
X_train = df[selected_features]
y_train = df['isFake']  # Assuming 'isFake' is the target label

# Train models
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)  # Now model learns with feature names

# Ensure models directory exists
os.makedirs("E:/project/models", exist_ok=True)

# Save model
with open("E:/project/models/random_forest.pkl", "wb") as file:
    pickle.dump(rf_model, file)

print(" Model trained and saved successfully with feature names!")
