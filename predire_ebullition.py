from pathlib import Path
import numpy as np
import joblib
import pandas as pd
from outils_descripteurs import preparer_mol, calculer

# Chargement du modèle et des paramètres d'entraînement
DOSSIER = Path(__file__).parent
donnees = joblib.load(DOSSIER / "modele_ebullition.joblib")
modele = donnees["modele"]
calibrateur = donnees.get("calibrateur")
colonnes = donnees["colonnes"]


def predire_temperature(smiles: str) -> dict:
    """
    Prédit la température d'ébullition à partir d'une chaîne SMILES.
    
    Paramètres :
        smiles (str): Représentation SMILES de la molécule.
        
    Retourne :
        dict: Dictionnaire contenant la prédiction en °C et K, ainsi que les métriques.
        
    Lève :
        ValueError: Si le SMILES est invalide ou si le calcul des descripteurs échoue.
    """
    mol = preparer_mol(smiles)
    if mol is None:
        raise ValueError("Code SMILES invalide ou impossible à standardiser.")

    # Calcul des descripteurs physico-chimiques
    valeurs_desc = calculer(mol)
    X = pd.DataFrame([valeurs_desc], columns=colonnes).replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        raise ValueError("Descripteurs chimiques incomplets ou invalides pour cette molécule.")

    # Prédiction avec le modèle de base
    prediction = float(modele.predict(X)[0])

    # Application du calibrateur (si disponible)
    if calibrateur is not None:
        prediction = float(calibrateur.predict([[prediction]])[0])

    prediction = round(prediction, 2)
    y_min = float(donnees.get("y_min", 0.0))
    y_max = float(donnees.get("y_max", 0.0))
    in_domain = bool(y_min <= prediction <= y_max)

    return {
        "prediction": prediction,
        "prediction_k": round(prediction + 273.15, 2),
        "mae": round(float(donnees.get("mae_test", 0.0)), 1),
        "pression_atm": float(donnees.get("pression_atm", 1.0)),
        "in_domain": in_domain,
        "y_min": y_min,
        "y_max": y_max,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Prédicteur de Température d'Ébullition (Mode Terminal)")
    print("=" * 60)
    while True:
        s = input("\nEntrez le SMILES (vide pour quitter) : ").strip()
        if not s:
            break

        try:
            res = predire_temperature(s)
            print(f"\nTempérature d'ébullition estimée : {res['prediction']:.2f} °C (± {res['mae']:.1f} °C en moyenne)")
            print(f"Température d'ébullition estimée : {res['prediction_k']:.2f} K")
            print(f"Pression de référence : {res['pression_atm']} atm")

            if not res["in_domain"]:
                print(f"Attention : hors de la plage d'entraînement ({res['y_min']:.0f} à {res['y_max']:.0f} °C), estimation peu fiable.")
        except Exception as e:
            print(f"Erreur : {e}")
