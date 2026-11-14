import sys
import os

# Ajouter le répertoire racine au PYTHONPATH pour importer app.py et charger les fichiers
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app

# Point d'entrée de la fonction serverless Vercel
app = app
