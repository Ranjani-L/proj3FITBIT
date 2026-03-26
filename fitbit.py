import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("C:\\Users\\Admin\\Desktop\\env\\Scripts\\Fitbit_dataset.csv")

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Workout_Type'] = df['Workout_Type'].map({
    'Cardio': 0, 'Strength': 1, 'HIIT': 2, 'Yoga': 3
})

X = df.drop(["Calories_Burned (kcal)", "Unnamed: 0"], axis=1)
y = df["Calories_Burned (kcal)"]
print(X.columns)
print(len(X.columns))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=300))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

pickle.dump(pipeline, open("pipeline.pkl", "wb"))

print("✅ Model saved successfully!")