from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Charger le modèle
model = pickle.load(open(os.path.join(os.path.dirname(__file__), "rf.pkl"), "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupération des données du formulaire
        credit_score = float(request.form['creditscore'])
        age = float(request.form['age'])
        tenure = int(request.form['tenure'])
        balance = float(request.form['balance'])
        num_products = int(request.form['numofproducts'])
        has_cr_card = int(request.form['hascrcard'])
        is_active = int(request.form['isactivemember'])
        est_salary = float(request.form['estimatedsalary'])
        gender = request.form['gender']
        geography = request.form['geography']

        # Encodage manuel (conforme à l'entraînement)
        gender_val = 1 if gender == 'Male' else 0
        geography_val = {'France': 0, 'Germany': 1, 'Spain': 2}[geography]

        # Création du tableau de caractéristiques
        features = np.array([[credit_score, geography_val, gender_val, age, tenure,
                              balance, num_products, has_cr_card, is_active, est_salary]])

        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return f"Erreur lors de la prédiction : {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

