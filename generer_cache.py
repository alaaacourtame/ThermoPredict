"""
generer_cache.py — Script de pré-calcul des descripteurs moléculaires (RDKit).
=============================================================================
À exécuter UNE SEULE FOIS en local (dans votre environnement avec rdkit installé) :

    python generer_cache.py

Ce script :
1. Parcourt le dictionnaire local COMMON_MOLECULES (Ethanol, Acétone, Benzène, etc.).
2. Calcule avec RDKit l'ensemble des descripteurs requis par le modèle Random Forest,
   dont notamment MolMR, Chi1, BertzCT, LabuteASA.
3. Sauvegarde les descripteurs, la formule et le SMILES dans 'cache_molecules.json'.
4. Ce fichier JSON permet au serveur cloud (Render / Enzonic 512 Mo) de tourner SANS RDKit.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ─── 1. Chargement du modèle pour obtenir la liste exacte des colonnes ────────
DOSSIER = Path(__file__).parent
MODELE_PATH = DOSSIER / "modele_ebullition.joblib"

if not MODELE_PATH.exists():
    raise FileNotFoundError(f"Modèle introuvable à l'emplacement : {MODELE_PATH}")

print(f"Chargement du modèle : {MODELE_PATH.name}...")
donnees = joblib.load(MODELE_PATH)
modele = donnees["modele"]
calibrateur = donnees.get("calibrateur")
colonnes = donnees["colonnes"]
print(f"-> Le modèle attend {len(colonnes)} descripteurs physico-chimiques.")

# ─── 2. Importation de RDKit (en local uniquement) ───────────────────────────
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
except ImportError as err:
    print("\n[ERREUR] RDKit n'est pas installé dans cet environnement Python.")
    print("Veuillez activer votre environnement Conda / venv avec RDKit, par exemple :")
    print("    conda activate mon_env")
    print("    python generer_cache.py\n")
    raise err

_chooser = rdMolStandardize.LargestFragmentChooser()
EXCLUS = {"Ipc", "AvgIpc"}
DESCRIPTEURS_RDKIT = [(n, f) for n, f in Descriptors._descList if n not in EXCLUS]


def preparer_molecule(smiles: str):
    """Convertit un SMILES en molécule RDKit standardisée."""
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    try:
        mol = _chooser.choose(mol)
    except Exception:
        pass
    return mol


def calculer_descripteurs(mol) -> dict:
    """Calcule tous les descripteurs RDKit et retourne un dictionnaire {nom: valeur}."""
    valeurs = {}
    for nom, fonction in DESCRIPTEURS_RDKIT:
        try:
            v = float(fonction(mol))
        except Exception:
            v = np.nan
        if not np.isfinite(v) or abs(v) > 1e9:
            v = np.nan
        valeurs[nom] = v
    return valeurs


# ─── 3. Dictionnaire des molécules usuelles ──────────────────────────────────
COMMON_MOLECULES = {
    # Eau
    "water":         {"smiles": "O",                             "formula": "H2O",       "name": "Water",        "cid": 962},
    "h2o":           {"smiles": "O",                             "formula": "H2O",       "name": "Water",        "cid": 962},
    "eau":           {"smiles": "O",                             "formula": "H2O",       "name": "Water",        "cid": 962},
    "o":             {"smiles": "O",                             "formula": "H2O",       "name": "Water",        "cid": 962},

    # Éthanol
    "ethanol":       {"smiles": "CCO",                           "formula": "C2H6O",     "name": "Ethanol",      "cid": 702},
    "c2h6o":         {"smiles": "CCO",                           "formula": "C2H6O",     "name": "Ethanol",      "cid": 702},
    "cco":           {"smiles": "CCO",                           "formula": "C2H6O",     "name": "Ethanol",      "cid": 702},

    # Benzène
    "benzene":       {"smiles": "c1ccccc1",                      "formula": "C6H6",      "name": "Benzene",      "cid": 241},
    "c6h6":          {"smiles": "c1ccccc1",                      "formula": "C6H6",      "name": "Benzene",      "cid": 241},

    # Acétone
    "acetone":       {"smiles": "CC(=O)C",                       "formula": "C3H6O",     "name": "Acetone",      "cid": 180},
    "c3h6o":         {"smiles": "CC(=O)C",                       "formula": "C3H6O",     "name": "Acetone",      "cid": 180},

    # Caféine
    "caffeine":      {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "formula": "C8H10N4O2", "name": "Caffeine",     "cid": 2519},
    "cafeine":       {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "formula": "C8H10N4O2", "name": "Caffeine",     "cid": 2519},
    "c8h10n4o2":     {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "formula": "C8H10N4O2", "name": "Caffeine",     "cid": 2519},

    # Acide acétique
    "acetic acid":   {"smiles": "CC(=O)O",                       "formula": "C2H4O2",    "name": "Acetic acid",  "cid": 176},
    "acide acetique":{"smiles": "CC(=O)O",                       "formula": "C2H4O2",    "name": "Acetic acid",  "cid": 176},
    "cc(=o)o":       {"smiles": "CC(=O)O",                       "formula": "C2H4O2",    "name": "Acetic acid",  "cid": 176},

    # Méthanol
    "methanol":      {"smiles": "CO",                            "formula": "CH4O",      "name": "Methanol",     "cid": 887},
    "ch4o":          {"smiles": "CO",                            "formula": "CH4O",      "name": "Methanol",     "cid": 887},

    # Méthane
    "methane":       {"smiles": "C",                             "formula": "CH4",       "name": "Methane",      "cid": 297},
    "ch4":           {"smiles": "C",                             "formula": "CH4",       "name": "Methane",      "cid": 297},
}

# ─── 4. Calcul et génération du cache JSON ────────────────────────────────────
def generer_cache():
    print(f"\nPré-calcul des descripteurs pour les molécules du dictionnaire...")
    
    # Étape A : Calculer par SMILES unique pour éviter les doublons de calcul
    smiles_uniques = {}
    for cle, data in COMMON_MOLECULES.items():
        s = data["smiles"]
        if s not in smiles_uniques:
            smiles_uniques[s] = data

    cache_par_smiles = {}

    for s, data in smiles_uniques.items():
        nom = data["name"]
        print(f"  -> Calcul pour {nom:15s} (SMILES: {s:10s}) ...", end=" ", flush=True)

        mol = preparer_molecule(s)
        if mol is None:
            print("ERREUR (SMILES non parsé)")
            continue

        desc = calculer_descripteurs(mol)

        # Filtrer et aligner sur les colonnes du modèle
        desc_modele = {}
        for c in colonnes:
            val = desc.get(c)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                desc_modele[c] = None
            else:
                desc_modele[c] = round(float(val), 6)

        # Vérification spécifique des facteurs XAI clés demandés
        facteurs_cles = ["MolMR", "Chi1", "BertzCT", "LabuteASA"]
        present_xai = {f: desc_modele.get(f) for f in facteurs_cles}

        # Test d'inférence directe pour vérifier la conformité du modèle
        row = [desc_modele.get(c) for c in colonnes]
        X = pd.DataFrame([row], columns=colonnes).replace([np.inf, -np.inf], np.nan)
        pred = float(modele.predict(X)[0])
        if calibrateur is not None:
            pred = float(calibrateur.predict([[pred]])[0])
        pred = round(pred, 2)

        cache_par_smiles[s] = {
            "smiles": s,
            "formula": data["formula"],
            "name": data["name"],
            "cid": data.get("cid"),
            "prediction_precalculee": pred,
            "descripteurs": desc_modele,
            "facteurs_cles": present_xai,
        }
        print(f"OK [T_eb: {pred:6.2f} °C]")

    # Étape B : Mapper toutes les clés (alias français, formules, minuscules)
    cache_final = {}
    for cle, data in COMMON_MOLECULES.items():
        s = data["smiles"]
        if s in cache_par_smiles:
            cache_final[cle] = cache_par_smiles[s]

    # Inclure aussi les SMILES directs (ex: "CCO", "O", "c1ccccc1")
    for s, item in cache_par_smiles.items():
        cache_final[s] = item
        cache_final[s.lower()] = item

    # Étape C : Écriture dans le fichier JSON
    destination = DOSSIER / "cache_molecules.json"
    with open(destination, "w", encoding="utf-8") as f:
        json.dump(cache_final, f, indent=2, ensure_ascii=False)

    print(f"\n" + "=" * 60)
    print(f"✅ Cache généré avec succès dans : {destination.name}")
    print(f"📊 {len(cache_par_smiles)} molécules uniques pré-calculées.")
    print(f"🔑 {len(cache_final)} clés de recherche mappées.")
    print(f"📦 Taille du fichier JSON : {destination.stat().st_size / 1024:.1f} Ko")
    print("=" * 60)
    print("\nVous pouvez maintenant lancer l'application en mode Cloud léger :")
    print("    python app.py\n")


if __name__ == "__main__":
    generer_cache()
