from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import sqlite3
import os
import json

app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_model.pkl")

# Initialize Database
def init_db():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    area REAL, bedrooms INTEGER, bathrooms INTEGER,
                    floors INTEGER, year_built INTEGER, location TEXT,
                    condition TEXT, garage TEXT, predicted_price REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

def generate_ai_insight(data, price):
    insight = "🧠 AI Analysis: "
    added = False
    
    if data['Location'] == 'Downtown' and int(data['Bedrooms']) >= 3:
        insight += "Large properties in Downtown are rare and highly sought after. "
        added = True
    elif data['Location'] == 'Suburban' and int(data['Area']) > 2000:
        insight += "Spacious suburban homes usually command strong market interest from families. "
        added = True

    if data['Condition'] in ['Poor', 'Fair']:
        insight += "Renovating the property to 'Good' or 'Excellent' condition could significantly increase its valuation. "
        added = True
    elif data['Condition'] == 'Excellent':
        insight += "The excellent condition of the property justifies a premium valuation. "
        added = True
        
    if data['Garage'] == 'No':
        insight += "Adding a garage space might boost market appeal. "
        added = True
        
    if not added:
        insight += "This property has a balanced set of features typical for its location. "
        
    return insight

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    ai_insight = None

    if request.method == "POST":
        try:
            area = float(request.form["Area"])
            bedrooms = int(request.form["Bedrooms"])
            bathrooms = int(request.form["Bathrooms"])
            floors = int(request.form["Floors"])
            yearbuilt = int(request.form["YearBuilt"])
            location = request.form["Location"]
            condition = request.form["Condition"]
            garage = request.form["Garage"]

            input_data = pd.DataFrame([{
                "Area": area,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Floors": floors,
                "YearBuilt": yearbuilt,
                "Location": location,
                "Condition": condition,
                "Garage": garage
            }])

            # Make Prediction
            predicted_price = model.predict(input_data)[0]
            prediction = round(float(predicted_price), 2)
            
            # Generate Insight
            ai_insight = generate_ai_insight(request.form, prediction)

            # Save to Database
            conn = sqlite3.connect("predictions.db")
            c = conn.cursor()
            c.execute('''INSERT INTO history 
                         (area, bedrooms, bathrooms, floors, year_built, location, condition, garage, predicted_price)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                      (area, bedrooms, bathrooms, floors, yearbuilt, location, condition, garage, prediction))
            conn.commit()
            conn.close()

        except Exception as e:
            error = f"Error: {str(e)}"

    # Fetch History
    conn = sqlite3.connect("predictions.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT 3")
    recent_predictions = [dict(row) for row in c.fetchall()]
    conn.close()
    
    # Load feature importances if available
    feature_importances = None
    if os.path.exists("feature_importances.json"):
        with open("feature_importances.json", "r") as f:
            feature_importances = json.dumps(json.load(f))

    return render_template("index.html", 
                           prediction=prediction, 
                           error=error,
                           history=recent_predictions,
                           feature_importances=feature_importances,
                           ai_insight=ai_insight)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    try:
        input_data = pd.DataFrame([data])
        price = model.predict(input_data)[0]
        return jsonify({"predicted_price": round(float(price), 2), "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400

if __name__ == "__main__":
    app.run(debug=True)