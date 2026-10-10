from flask import Flask, jsonify, render_template, request
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

app = Flask(__name__)

df = pd.read_csv(Path(__file__).with_name("data") / "RUL_prediction_dataset.csv")
target = "RUL_Cycles"

X = df.drop(columns=[target])
y = df[target]
categorical_features = ["Equipment_ID"]
numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numerical", "passthrough", numerical_features),
    ]
)
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

joblib.dump(model, Path(__file__).with_name("rul_model.pkl"))


@app.route("/", methods=["GET"])
def home():
    sample = X.iloc[0].to_dict()
    features = [
        {
            "name": column,
            "label": column.replace("_", " "),
            "value": sample[column],
            "type": "text" if column == "Equipment_ID" else "number",
        }
        for column in X.columns
    ]
    return render_template("index.html", features=features)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "message": "Predictive Maintenance RUL API is running",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({
                "error": "Please send a JSON object"
            }), 400

        # Convert input into a DataFrame
        input_df = pd.DataFrame([data])

        # Check required feature names
        missing = [
            col for col in X.columns
            if col not in input_df.columns
        ]

        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing
            }), 400

        # Keep the same feature order used in training
        input_df = input_df[X.columns]

        # Predict RUL
        prediction = model.predict(input_df)[0]

        return jsonify({
            "predicted_RUL_cycles": round(
                float(prediction), 2
            )
        })

    except (ValueError, TypeError) as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
