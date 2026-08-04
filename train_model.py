import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("House Price Prediction Dataset.csv")

# Drop unwanted column
df = df.drop(columns=["Id"])

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Separate column types
categorical_cols = ["Location", "Condition", "Garage"]
numerical_cols = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning (GridSearchCV)
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
}

print("Running Grid Search CV...")
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model trained successfully")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.4f}")

# Extract Feature Importances
cat_encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
cat_features = cat_encoder.get_feature_names_out(categorical_cols)
feature_names = list(cat_features) + numerical_cols

importances = model.named_steps["regressor"].feature_importances_
importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}

# Save feature importances
with open("feature_importances.json", "w") as f:
    json.dump(importance_dict, f, indent=4)
print("Feature importances saved as feature_importances.json")

# Save model
joblib.dump(model, "house_price_model.pkl")
print("Model saved as house_price_model.pkl")