from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('car_eval_best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = {
            'buying': request.form['buying'],
            'maint': request.form['maint'],
            'doors': request.form['doors'],
            'persons': request.form['persons'],
            'lug_boot': request.form['lug_boot'],
            'safety': request.form['safety']
        }

        df = pd.DataFrame([features])
        df_encoded = pd.get_dummies(df)

        model_cols = model.feature_names_in_
        for col in model_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[model_cols]

        prediction = model.predict(df_encoded)[0]
        return render_template('index.html', prediction=f"Predicted Car Class: {prediction}")

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
