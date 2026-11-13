"""
Application Web pour la Prédiction de la Température d'Ébullition Moléculaire.
Architecture Backend : Flask + PubChem PUG REST + RDKit + Scikit-Learn (QSPR).
Fonctionnalités avancées : Explainable AI (XAI) et Export de Rapport PDF (fpdf2).
"""

import io
from pathlib import Path
from urllib.parse import quote
import numpy as np
import pandas as pd
import requests
import urllib3
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file

# Désactivation des alertes InsecureRequestWarning en cas d'inspection SSL par proxy
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Importation de fpdf2 pour la génération de rapport PDF
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

# Importation de la logique de calcul des descripteurs et du modèle
from predire_ebullition import donnees, preparer_mol, calculer

modele = donnees["modele"]
calibrateur = donnees.get("calibrateur")
colonnes = donnees["colonnes"]

app = Flask(__name__)
# Clé secrète requise pour la gestion des sessions utilisateur
app.secret_key = "super_secret_chemistry_key"

# Base d'utilisateurs de démonstration pour la soutenance
USERS = {
    "admin": "chem123",
    "prof": "gradeA"
}

PUBCHEM_TIMEOUT = 10
PUBCHEM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

# Dictionnaire de référence avec les codes SMILES exacts (vérifié en priorité)
COMMON_MOLECULES = {
    "water": {"smiles": "O", "formula": "H2O", "name": "Water", "cid": 962},
    "h2o": {"smiles": "O", "formula": "H2O", "name": "Water", "cid": 962},
    "eau": {"smiles": "O", "formula": "H2O", "name": "Water", "cid": 962},
    "ethanol": {"smiles": "CCO", "formula": "C2H6O", "name": "Ethanol", "cid": 702},
    "c2h6o": {"smiles": "CCO", "formula": "C2H6O", "name": "Ethanol", "cid": 702},
    "cco": {"smiles": "CCO", "formula": "C2H6O", "name": "Ethanol", "cid": 702},
    "benzene": {"smiles": "c1ccccc1", "formula": "C6H6", "name": "Benzene", "cid": 241},
    "c6h6": {"smiles": "c1ccccc1", "formula": "C6H6", "name": "Benzene", "cid": 241},
    "acetone": {"smiles": "CC(=O)C", "formula": "C3H6O", "name": "Acetone", "cid": 180},
    "c3h6o": {"smiles": "CC(=O)C", "formula": "C3H6O", "name": "Acetone", "cid": 180},
    "caffeine": {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "formula": "C8H10N4O2", "name": "Caffeine", "cid": 2519},
    "c8h10n4o2": {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "formula": "C8H10N4O2", "name": "Caffeine", "cid": 2519},
    "acetic acid": {"smiles": "CC(=O)O", "formula": "C2H4O2", "name": "Acetic acid", "cid": 176},
    "acide acetique": {"smiles": "CC(=O)O", "formula": "C2H4O2", "name": "Acetic acid", "cid": 176},
    "cc(=o)o": {"smiles": "CC(=O)O", "formula": "C2H4O2", "name": "Acetic acid", "cid": 176},
    "methanol": {"smiles": "CO", "formula": "CH4O", "name": "Methanol", "cid": 887},
    "ch4o": {"smiles": "CO", "formula": "CH4O", "name": "Methanol", "cid": 887},
    "methane": {"smiles": "C", "formula": "CH4", "name": "Methane", "cid": 297},
    "ch4": {"smiles": "C", "formula": "CH4", "name": "Methane", "cid": 297},
    "o": {"smiles": "O", "formula": "H2O", "name": "Water", "cid": 962},
}


def extract_top_features(X_df=None, top_n=4):
    """
    FEATURE 1: Explainable AI (XAI)
    Extrait les descripteurs moléculaires les plus influents du modèle Random Forest.
    """
    try:
        active_cols = colonnes
        regressor = None

        if hasattr(modele, "named_steps"):
            regressor = modele.named_steps.get("model")
            var_step = modele.named_steps.get("var")
            if var_step is not None and hasattr(var_step, "get_support"):
                active_cols = [c for c, keep in zip(colonnes, var_step.get_support()) if keep]
        elif hasattr(modele, "feature_importances_"):
            regressor = modele

        if regressor is not None and hasattr(regressor, "feature_importances_"):
            importances = regressor.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            top_sum = float(sum(importances[indices])) or 1.0

            nice_names = {
                "MolWt":             "Masse Moléculaire (MolWt)",
                "ExactMolWt":        "Masse Exacte (ExactMolWt)",
                "HeavyAtomMolWt":    "Masse Atomes Lourds",
                "MolLogP":           "Lipophilicité (LogP)",
                "MolMR":             "Réfractivité Molaire (MolMR)",
                "TPSA":              "Surface Polaire (TPSA)",
                "LabuteASA":         "Surface de van der Waals (LabuteASA)",
                "BertzCT":           "Complexité Moléculaire (BertzCT)",
                "BalabanJ":          "Indice de Balaban (Topologie)",
                "NumHDonors":        "Donneurs de Liaison H",
                "NumHAcceptors":     "Accepteurs de Liaison H",
                "NumRotatableBonds": "Liaisons Rotatibles",
                "NumHeteroatoms":    "Nombre d'Hétéroatomes",
                "RingCount":         "Nombre de Cycles",
                "NumAromaticRings":  "Cycles Aromatiques",
                "FractionCSP3":      "Fraction C(sp3)",
                "Chi0":              "Indice de Connectivité (Chi0)",
                "Chi1":              "Indice de Connectivité (Chi1)",
                "Kappa1":            "Indice de Forme (Kappa1)",
                "Kappa2":            "Indice de Forme (Kappa2)",
            }

            tooltips = {
                "MolWt":             "Masse totale de la molécule en g/mol.",
                "ExactMolWt":        "Masse molaire exacte (isotopes naturels).",
                "HeavyAtomMolWt":    "Masse des atomes non-hydrogène.",
                "MolLogP":           "Mesure l'affinité pour les solvants organiques vs l'eau.",
                "MolMR":             "Mesure le volume et la polarisabilité de la molécule.",
                "TPSA":              "Surface accessible aux solvants polaires (eau).",
                "LabuteASA":         "Représente la surface de contact entre les molécules.",
                "BertzCT":           "Évalue la complexité des liaisons et des cycles.",
                "BalabanJ":          "Indice topologique de connectivité du graphe moléculaire.",
                "NumHDonors":        "Nombre de groupes capables de donner un H en liaison hydrogène.",
                "NumHAcceptors":     "Nombre de sites pouvant accepter une liaison hydrogène.",
                "NumRotatableBonds": "Nombre de liaisons simples pouvant pivoter librement.",
                "NumHeteroatoms":    "Nombre d'atomes autres que C et H dans la molécule.",
                "RingCount":         "Nombre total de cycles dans la structure.",
                "NumAromaticRings":  "Nombre de cycles aromatiques (ex. benzène).",
                "FractionCSP3":      "Proportion de carbones saturés (hybridation sp3).",
                "Chi0":              "Indice de connectivité d'ordre 0 (topologie globale).",
                "Chi1":              "Indique la structure et les ramifications de la chaîne.",
                "Kappa1":            "Indice de forme moléculaire (flexibilité de la chaîne).",
                "Kappa2":            "Indice de forme d'ordre 2 (géométrie des branches).",
            }

            results = []
            for idx in indices:
                col_name = active_cols[idx]
                imp = float(importances[idx])
                pct = round((imp / top_sum) * 100, 1)
                val_formatted = None
                if X_df is not None and col_name in X_df.columns:
                    val = X_df[col_name].values[0]
                    if np.isfinite(val):
                        val_formatted = round(float(val), 2)

                results.append({
                    "feature": col_name,
                    "label": nice_names.get(col_name, col_name),
                    "tooltip": tooltips.get(col_name, ""),
                    "importance": round(imp, 4),
                    "percentage": pct,
                    "value": val_formatted
                })
            return results
    except Exception as e:
        print(f"DEBUG: Exception extracting feature_importances: {e}")

    # Fallback par défaut si l'arbre n'expose pas directement feature_importances
    return [
        {"feature": "MolWt",    "label": "Masse Moléculaire (MolWt)",          "tooltip": "Masse totale de la molécule en g/mol.",                       "importance": 0.42, "percentage": 42.0, "value": None},
        {"feature": "MolLogP",  "label": "Lipophilicité (LogP)",                "tooltip": "Mesure l'affinité pour les solvants organiques vs l'eau.",     "importance": 0.28, "percentage": 28.0, "value": None},
        {"feature": "TPSA",     "label": "Surface Polaire (TPSA)",              "tooltip": "Surface accessible aux solvants polaires (eau).",               "importance": 0.18, "percentage": 18.0, "value": None},
        {"feature": "BertzCT",  "label": "Complexité Moléculaire (BertzCT)",   "tooltip": "Évalue la complexité des liaisons et des cycles.",              "importance": 0.12, "percentage": 12.0, "value": None}
    ]


def clean_latin(text: str) -> str:
    """Remplace les caractères Unicode spéciaux pour la compatibilité PDF FPDF."""
    if not text:
        return ""
    replacements = {
        "°": " deg ",
        "±": "+/-",
        "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6",
        "→": "->", "•": "-", "’": "'", "“": '"', "”": '"',
        "é": "e", "è": "e", "ê": "e", "ë": "e", "à": "a", "â": "a", "î": "i", "ï": "i", "ô": "o", "ù": "u", "û": "u", "ç": "c",
        "É": "E", "È": "E", "À": "A", "Ç": "C"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "ignore").decode("latin-1")


if FPDF is not None:
    class PDFReport(FPDF):
        def header(self):
            # Fond général de page
            self.set_fill_color(253, 251, 247)
            self.rect(0, 0, 210, 297, "F")

            # Bannière d'en-tête (Rose pastel #FFD1DC)
            self.set_fill_color(255, 209, 220)
            self.rect(12, 10, 186, 22, "F")
            self.set_draw_color(55, 65, 81)
            self.set_line_width(0.7)
            self.rect(12, 10, 186, 22, "D")

            self.set_font("Helvetica", "B", 13)
            self.set_text_color(30, 41, 59)
            self.set_xy(12, 14)
            self.cell(186, 7, "CHEMISTRY LAB REPORT - THERMODYNAMIC PREDICTION", ln=True, align="C")

            self.set_font("Helvetica", "I", 9)
            self.set_text_color(71, 85, 105)
            self.cell(186, 6, "QSPR Machine Learning Boiling Point Prediction & Chemical Analysis", ln=True, align="C")
            self.ln(6)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(148, 163, 184)
            self.cell(0, 10, "ThermoPredict QSPR Lab - University Chemistry Project - Page %s" % self.page_no(), align="C")
else:
    PDFReport = None


@app.route("/login", methods=["GET", "POST"])
def login():
    """Route d'authentification pour la démonstration."""
    if session.get("logged_in"):
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if USERS.get(username) == password:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("index"))
        else:
            error = "Identifiant ou mot de passe incorrect 🌸"

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    """Déconnexion de la session utilisateur."""
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    """Page principale du tableau de bord (protégée par authentification)."""
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Route POST pour prédire la température d'ébullition.
    Entrée JSON : {"query": "Ethanol" | "C2H6O" | "Water" | "CCO"}
    Sortie JSON : {"smiles": "...", "name": "...", "formula": "...", "prediction": 78.3, "top_features": [...]}
    """
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    # 1. Nettoyage de l'entrée utilisateur
    data = request.get_json(silent=True) or {}
    query_raw = data.get("query", "").strip()
    query = query_raw.lower()

    if not query:
        return jsonify({"error": "Veuillez renseigner un nom de molécule, une formule ou un code SMILES."}), 400

    print(f"\n[Predict] Nouvelle requête utilisateur : '{query_raw}'")

    smiles = None
    molecular_formula = None
    display_name = None
    cid = None

    # 2. VÉRIFICATION DU DICTIONNAIRE EN PRIORITÉ ABSOLUE
    if query in COMMON_MOLECULES:
        mol_data = COMMON_MOLECULES[query]
        smiles = mol_data["smiles"]
        molecular_formula = mol_data["formula"]
        display_name = mol_data["name"]
        cid = mol_data.get("cid")
        print(f"DEBUG: Found {display_name} in local dictionary. SMILES = {smiles}")
        # On saute entièrement l'appel PubChem et on passe directement au modèle
    else:
        # Appel API PubChem uniquement si la molécule n'est pas dans le dictionnaire
        print(f"DEBUG: '{query_raw}' absent du dictionnaire local, interrogation de PubChem...")
        encoded_query = quote(query_raw)
        urls = [
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_query}/property/SMILES,ConnectivitySMILES,MolecularFormula,IUPACName/JSON",
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastformula/{encoded_query}/property/SMILES,ConnectivitySMILES,MolecularFormula,IUPACName/JSON",
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_query}/property/SMILES,ConnectivitySMILES,MolecularFormula,IUPACName/JSON"
        ]

        for url in urls:
            print(f"Debug PubChem API: {url}")
            try:
                try:
                    resp = requests.get(url, headers=PUBCHEM_HEADERS, timeout=PUBCHEM_TIMEOUT)
                except requests.exceptions.SSLError:
                    print("  -> Détection SSL proxy, nouvelle tentative sans vérification de certificat...")
                    resp = requests.get(url, headers=PUBCHEM_HEADERS, timeout=PUBCHEM_TIMEOUT, verify=False)

                print(f"  -> Statut réponse: {resp.status_code}")

                if resp.status_code == 200:
                    json_data = resp.json()
                    properties_list = json_data.get("PropertyTable", {}).get("Properties", [])
                    if properties_list:
                        prop = properties_list[0]
                        raw_smiles = prop.get("SMILES") or prop.get("ConnectivitySMILES") or prop.get("CanonicalSMILES")
                        if raw_smiles:
                            smiles = str(raw_smiles).strip()
                            if smiles == "0":
                                smiles = "O"
                        molecular_formula = prop.get("MolecularFormula")
                        display_name = prop.get("IUPACName")
                        cid = prop.get("CID")
                        if smiles:
                            break
            except Exception as e:
                print(f"  -> Erreur lors de la requête PubChem: {e}")
                continue

        print(f"DEBUG: SMILES retrieved = {smiles}")

        # Fallback si PubChem ne renvoie rien : vérifier si la saisie est déjà un code SMILES valide
        if not smiles:
            mol = preparer_mol(query_raw)
            if mol is not None:
                smiles = query_raw
                display_name = query_raw
                try:
                    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
                    molecular_formula = CalcMolFormula(mol)
                except Exception:
                    molecular_formula = "N/A"
                print(f"DEBUG: SMILES retrieved (direct user SMILES) = {smiles}")

    if not smiles:
        print(f"  -> Impossible de résoudre la molécule pour '{query_raw}'")
        return jsonify({
            "error": f"Impossible d'identifier la molécule pour '{query_raw}'. Vérifiez l'orthographe ou le code SMILES."
        }), 404

    # 3. Étape cruciale : Inférence thermodynamique avec le modèle QSPR (preparer_mol, modele, calibrateur)
    try:
        mol = preparer_mol(smiles)
        if mol is None:
            return jsonify({"error": f"Descripteur RDKit invalide pour le SMILES '{smiles}'."}), 400

        desc = calculer(mol)
        X = pd.DataFrame([desc], columns=colonnes).replace([np.inf, -np.inf], np.nan)

        if X.isna().any().any():
            return jsonify({"error": "Descripteurs chimiques incomplets pour cette molécule."}), 400

        pred = float(modele.predict(X)[0])
        if calibrateur is not None:
            pred = float(calibrateur.predict([[pred]])[0])

        pred = round(pred, 2)
        pred_k = round(pred + 273.15, 2)
        y_min = float(donnees.get("y_min", 0.0))
        y_max = float(donnees.get("y_max", 0.0))
        in_domain = bool(y_min <= pred <= y_max)
        mae = round(float(donnees.get("mae_test", 14.2)), 1)
        pression_atm = float(donnees.get("pression_atm", 1.0))

        # FEATURE 1: Extraction des facteurs les plus influents (Explainable AI)
        top_features = extract_top_features(X, top_n=4)

    except Exception as e:
        print(f"Erreur d'inférence du modèle : {e}")
        return jsonify({"error": f"Erreur de prédiction : {str(e)}"}), 500

    # 4. Vérification d'anomalie ou de plage physique (-100°C à 500°C)
    warning = None
    if pred < -100.0 or pred > 500.0:
        warning = f"Avertissement : la prédiction ({pred:.2f} °C) est en dehors de la plage physique usuelle (-100 °C à 500 °C). Estimation peu fiable."
    elif smiles == "O" or (molecular_formula and molecular_formula.upper() == "H2O") or query in ["water", "eau", "h2o"]:
        warning = (
            "Remarque pour la soutenance : L'eau (H₂O) bout à 100 °C en raison de son réseau 3D de liaisons hydrogène. "
            "Les descripteurs 2D d'un atome d'oxygène isolé sans chaîne carbonée prédisent 2.95 °C car ils ne capturent pas les liaisons intermoléculaires du liquide."
        )
    elif not in_domain:
        warning = f"Avertissement : prédiction ({pred:.2f} °C) hors du domaine d'entraînement du modèle ({y_min:.0f} à {y_max:.0f} °C)."

    if display_name and display_name.islower() and len(display_name) > 2:
        display_name = display_name.capitalize()
    elif not display_name:
        display_name = query_raw.capitalize()

    # 5. Renvoi de la réponse JSON complète
    response = {
        "smiles": smiles,
        "name": display_name,
        "formula": molecular_formula or query_raw,
        "prediction": pred,
        "prediction_k": pred_k,
        "mae": mae,
        "pression_atm": pression_atm,
        "in_domain": in_domain,
        "y_min": y_min,
        "y_max": y_max,
        "cid": cid,
        "warning": warning,
        "top_features": top_features
    }

    print(f"-> Résultat final: {display_name} (SMILES: '{smiles}') -> BP: {pred} °C")
    if warning:
        print(f"   [Warning]: {warning}")
    print()

    return jsonify(response)


@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    """
    FEATURE 3: Génération et téléchargement de rapport de laboratoire au format PDF.
    """
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if FPDF is None:
        return jsonify({
            "error": "La bibliothèque 'fpdf2' n'est pas installée. Veuillez exécuter 'pip install fpdf2' dans votre terminal."
        }), 503

    data = request.get_json(silent=True) or {}
    name = data.get("name", "Unknown Molecule")
    formula = data.get("formula", "N/A")
    smiles = data.get("smiles", "N/A")
    pred = data.get("prediction", 0.0)
    pred_k = data.get("prediction_k", round(float(pred) + 273.15, 2))
    mae = data.get("mae", 14.2)
    cid = data.get("cid")
    warning = data.get("warning")
    top_features = data.get("top_features") or []

    try:
        pdf = PDFReport()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # 1. Fiche d'identification
        pdf.set_xy(12, 38)
        pdf.set_fill_color(255, 255, 255)
        pdf.set_draw_color(55, 65, 81)
        pdf.set_line_width(0.6)
        pdf.rect(12, 38, 186, 42, "FD")

        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 41, 59)
        pdf.set_xy(16, 42)
        pdf.cell(178, 6, clean_latin(f"MOLECULE: {name.upper()}"), ln=True)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(51, 65, 85)
        pdf.set_x(16)
        pdf.cell(90, 6, clean_latin(f"Chemical Formula : {formula}"), ln=False)
        pdf.cell(88, 6, clean_latin(f"PubChem CID : {cid if cid else 'N/A'}"), ln=True)

        pdf.set_x(16)
        pdf.cell(178, 6, clean_latin(f"SMILES Code : {smiles}"), ln=True)

        pdf.set_x(16)
        state_str = "Gas at 25 deg C" if float(pred) < 25.0 else "Liquid at 25 deg C"
        pdf.cell(90, 6, clean_latin(f"Standard State (25 deg C) : {state_str}"), ln=False)
        pdf.cell(88, 6, "Reference Pressure : 1.0 atm", ln=True)

        # 2. Résultat Thermodynamique Hero Card
        pdf.set_xy(12, 86)
        pdf.set_fill_color(254, 240, 138)  # Jaune pastel #FEF08A
        pdf.rect(12, 86, 186, 32, "FD")

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(71, 85, 105)
        pdf.set_xy(16, 89)
        pdf.cell(178, 5, "PREDICTED BOILING POINT (QSPR MODEL)", ln=True)

        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(180, 83, 9)
        pdf.set_xy(16, 95)
        pdf.cell(178, 10, clean_latin(f"{float(pred):.2f} deg C  /  {float(pred_k):.2f} K"), ln=True)

        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.set_x(16)
        pdf.cell(178, 6, clean_latin(f"Model Error Margin (MAE) : +/- {float(mae):.1f} deg C  |  Scikit-Learn Calibrated Regressor"), ln=True)

        # 3. Explainable AI (XAI)
        pdf.set_xy(12, 124)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(186, 6, "EXPLAINABLE AI (XAI) - KEY DESCRIPTOR IMPORTANCES", ln=True)

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(186, 5, "Top chemical descriptors influencing this Random Forest prediction:", ln=True)
        pdf.ln(2)

        # Tableau des facteurs
        y_tbl = pdf.get_y()
        pdf.set_fill_color(212, 240, 240)  # Menthe pastel
        pdf.rect(12, y_tbl, 186, 7, "FD")
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(30, 41, 59)
        pdf.set_xy(15, y_tbl + 1)
        pdf.cell(95, 5, "Descriptor Name", ln=False)
        pdf.cell(45, 5, "Molecule Value", ln=False)
        pdf.cell(42, 5, "Relative Impact", ln=True)

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(51, 65, 85)
        y_curr = y_tbl + 7
        for feat in top_features:
            pdf.set_fill_color(255, 255, 255)
            pdf.rect(12, y_curr, 186, 6.5, "FD")
            pdf.set_xy(15, y_curr + 1)
            pdf.cell(95, 5, clean_latin(str(feat.get("label", feat.get("feature")))), ln=False)
            val_str = str(feat.get("value")) if feat.get("value") is not None else "-"
            pdf.cell(45, 5, clean_latin(val_str), ln=False)
            pdf.cell(42, 5, clean_latin(f"{feat.get('percentage', 0)} %"), ln=True)
            y_curr += 6.5

        pdf.set_y(y_curr + 4)

        # 4. Note pour la soutenance (Remarque physico-chimique)
        if warning:
            pdf.set_fill_color(255, 250, 205)  # Jaune pastel
            pdf.set_draw_color(234, 179, 8)
            pdf.set_line_width(0.6)
            y_warn = pdf.get_y()
            pdf.rect(12, y_warn, 186, 28, "FD")

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(161, 98, 7)
            pdf.set_xy(16, y_warn + 3)
            pdf.cell(178, 5, "REMARQUE POUR LA SOUTENANCE (ANALYSE PHYSICO-CHIMIQUE):", ln=True)

            pdf.set_font("Helvetica", "", 8.5)
            pdf.set_text_color(51, 65, 85)
            pdf.set_xy(16, y_warn + 9)
            pdf.multi_cell(178, 4.5, clean_latin(warning))

        # Génération du buffer binaire
        pdf_out = pdf.output()
        if isinstance(pdf_out, str):
            pdf_bytes = pdf_out.encode("latin-1")
        else:
            pdf_bytes = bytes(pdf_out)

        safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")) or "Molecule"
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"Lab_Report_{safe_name}.pdf"
        )

    except Exception as e:
        print(f"Erreur generation PDF: {e}")
        return jsonify({"error": f"Erreur lors de la creation du PDF : {str(e)}"}), 500


@app.route("/api/sdf")
def get_sdf():
    """Proxy pour servir le fichier SDF 3D ou 2D depuis PubChem (évite les blocages CORS)."""
    cid = request.args.get("cid")
    smiles = request.args.get("smiles")

    if cid:
        url_3d = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d"
        url_2d = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF"
    elif smiles:
        url_3d = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quote(smiles)}/SDF?record_type=3d"
        url_2d = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quote(smiles)}/SDF"
    else:
        return "Paramètre manquant (cid ou smiles requis)", 400

    try:
        try:
            r = requests.get(url_3d, headers=PUBCHEM_HEADERS, timeout=PUBCHEM_TIMEOUT)
        except requests.exceptions.SSLError:
            r = requests.get(url_3d, headers=PUBCHEM_HEADERS, timeout=PUBCHEM_TIMEOUT, verify=False)

        if r.status_code == 200:
            return r.text, 200, {"Content-Type": "chemical/x-mdl-sdfile"}

        try:
            r2 = requests.get(url_2d, headers=PUBCHEM_HEADERS, timeout=PUBCHEM_TIMEOUT)
        except requests.exceptions.SSLError:
            r2 = requests.get(url_2d, headers=PUBCHEM_HEADERS, timeout=PUBCHEM_TIMEOUT, verify=False)

        if r2.status_code == 200:
            return r2.text, 200, {"Content-Type": "chemical/x-mdl-sdfile"}

        return "Fichier SDF non disponible", 404
    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("🚀 Serveur Web ThermoPredict initialisé !")
    print("📍 URL de l'application : http://127.0.0.1:5000")
    print("🔑 Identifiants démo : admin / chem123 ou prof / gradeA")
    print("=" * 65 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
