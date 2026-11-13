# ThermoPredict • Prédiction du Point d'Ébullition Moléculaire (QSPR)

Application Web interactive pour la prédiction de la température d'ébullition de composés chimiques par Machine Learning (QSPR avec RDKit, Scikit-Learn et PubChem).

---

## 🌟 Fonctionnalités Clés pour la Soutenance

1. **Recherche & Résolution Multi-Critères** :
   - Recherche par **Nom** (*Ethanol, Benzène, Acétone*), **Formule brute** (*C2H6O, H2O, C8H10N4O2*) ou **SMILES** direct (*CCO, O*).
   - Dictionnaire local prioritaire pour une réactivité instantanée + fallback automatique sur l'API PubChem PUG REST.

2. **Inférence QSPR & Calibration** :
   - Calcul en temps réel de 200+ descripteurs physico-chimiques RDKit.
   - Régression Random Forest / ExtraTrees avec calibration linéaire d'aplatissement de moyenne.

3. **Visualisation Moléculaire 3D (3Dmol.js)** :
   - Visualisation 3D interactive avec palette de couleurs pastel douces (Carbone violet, Oxygène rose, Hydrogène blanc).
   - Rotation continue, zoom, orbite et basculement de représentations (Bâtonnets, Sphères, Boules & Bâtons).
   - Support dédié pour la molécule d'eau ($H_2O$) via coordonnées XYZ explicites.

4. **✨ FEATURE 1 : Explainable AI (XAI - Feature Importance)** :
   - Extraction dynamique des 4 descripteurs moléculaires les plus influents (`feature_importances_` du Random Forest).
   - Barres de progression pastel affichant l'impact relatif (%) et la valeur du descripteur pour la molécule analysée.

5. **✨ FEATURE 2 : Mode Comparatif ("Compare Mode" ⚖️)** :
   - Bouton **"Add to Compare ⚖️"** permettant d'empiler jusqu'à 4 molécules dans un plateau comparatif (*Comparison Lab Tray*).
   - Comparaison côte à côte des températures d'ébullition et calcul automatique du différentiel thermique ($\Delta T$) par rapport à la molécule de référence.

6. **✨ FEATURE 3 : Export de Rapport de Laboratoire en PDF (fpdf2 📄)** :
   - Bouton **"Export as PDF 📄"** qui génère un document officiel au format PDF.
   - Inclut : Nom, Formule, SMILES, CID, Température (°C et K), Marge d'erreur (MAE), Tableau Explainable AI et Remarques physico-chimiques.

7. **Sécurité & Authentification Démo** :
   - Page de connexion dédiée : `admin` / `chem123` ou `prof` / `gradeA`.
   - Bouton d'autofill pour une démonstration fluide lors de la présentation.

---

## 🚀 Démarrage Rapide

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

*(Nécessite `flask`, `requests`, `fpdf2`, `rdkit`, `pandas`, `numpy`, `scikit-learn`, `joblib`)*

### 2. Lancement du serveur Web Flask

```bash
python app.py
```

Accédez ensuite à l'application dans votre navigateur :
👉 **http://127.0.0.1:5000**
