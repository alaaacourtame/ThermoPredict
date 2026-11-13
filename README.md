---

## 🧠 Approche Scientifique et Méthodologie

Ce projet ne se contente pas d'utiliser une formule empirique classique ; il repose sur une approche moderne de **QSPR** (Quantitative Structure-Property Relationship) couplée au Machine Learning.

### 1. Comparaison avec les méthodes classiques (Joback, Antoine)
Pour prédire une température d'ébullition, la chimie traditionnelle utilise des méthodes empiriques :
*   **La méthode de Joback-Reid :** Elle est basée sur la "contribution de groupes". La molécule est découpée en sous-groupes (ex: -CH3, -OH) et on additionne l'impact de chaque groupe via une formule fixe. Bien qu'efficace, elle est limitée aux groupes répertoriés et ne capture pas les interactions 3D complexes.
*   **L'équation d'Antoine :** Elle utilise des constantes (A, B, C) spécifiques à chaque molécule pour relier la pression de vapeur à la température. Elle nécessite une base de données gigantesque de constantes pré-calculées et ne permet pas de prédire la température d'une nouvelle molécule inconnue sans données préalables.

**L'avantage de l'approche QSPR (ce projet) :** Au lieu d'utiliser des formules figées, notre modèle **Random Forest** apprend automatiquement les relations complexes entre la structure mathématique de la molécule (les descripteurs) et sa température d'ébullition. Il peut ainsi prédire la température de n'importe quelle nouvelle molécule à partir de sa structure.

### 2. L'explication du code : Comment ça marche ?
Le pipeline de prédiction (dans `predire_ebullition.py` et `app.py`) suit les étapes suivantes :

1.  **Acquisition du SMILES :** L'utilisateur entre un nom, une formule ou un SMILES. Le backend interroge l'API PubChem pour obtenir le code SMILES canonique (la structure exacte de la molécule).
2.  **Vectorisation (RDKit) :** Le code SMILES est transformé en objet molécule via la bibliothèque `RDKit`. Le script calcule ensuite plus de 200 **descripteurs physico-chimiques** (MolMR, indices de connectivité, surface de van der Waals, etc.). Ces descripteurs sont stockés dans un DataFrame `Pandas`.
3.  **Inférence (Scikit-Learn) :** Ces données sont passées à un modèle **Random Forest** (chargé via `Joblib`). Le modèle génère une prédiction brute.
4.  **Calibration :** Un modèle de calibration linéaire ajuste la prédiction pour corriger les biais du Random Forest et améliorer la précision (erreur MAE réduite).
5.  **IA Explicable (Explainable AI) :** Le backend extrait les `feature_importances_` du modèle Random Forest. Cela permet d'afficher sur le frontend les 4 descripteurs qui ont le plus influencé la prédiction, rendant l'IA transparente.

### 3. Le rôle des bibliothèques utilisées
*   **`RDKit`** : Le cœur chimique du projet. Il convertit le texte (SMILES) en structure moléculaire et calcule les descripteurs mathématiques.
*   **`Scikit-Learn`** : Utilisé pour entraîner le modèle de régression (Random Forest) et le calibrateur.
*   **`Pandas` & `NumPy`** : Gèrent les matrices de données (les descripteurs) avant de les envoyer au modèle.
*   **`Flask`** : Le framework web backend qui fait le lien entre l'interface utilisateur et le modèle Python.
*   **`Joblib`** : Permet de sauvegarder et recharger le modèle entraîné (`.joblib`) sans avoir à le ré-entraîner à chaque démarrage du serveur.
*   **`3Dmol.js`** (Frontend) : Bibliothèque JavaScript qui récupère les coordonnées 3D de PubChem et dessine la molécule de manière interactive sur la page web.