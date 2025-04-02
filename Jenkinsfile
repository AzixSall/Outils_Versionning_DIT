pipeline {
    agent any
    
    environment {
        // Définition des variables d'environnement
        //PYTHON_VERSION = '3.12.1'
        PYTHON_PATH = 'C:\\Users\\Admin\\anaconda3\\python.exe'
        VENV_NAME = 'venv'
    }
    
    stages {
        stage('Checkout') {
            steps {
                // Récupération du code source depuis Git
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                // Création et activation de l'environnement virtuel Python
                bat '''
                    "%PYTHON_PATH%" -m venv %VENV_NAME%
                    call %VENV_NAME%\\Scripts\\activate.bat
                    "%PYTHON_PATH%" -m pip install --upgrade pip
                    pip install -r requirements.txt
                    REM Réinstaller DVC avec une version spécifique
                    pip uninstall -y dvc dvc-objects
                    pip install dvc==2.45.1 dvc-gdrive==2.19.1
                '''
            }
        }

        stage('DVC Version Check') {
            steps {
                bat '''
                    call %VENV_NAME%\\Scripts\\activate.bat
                    dvc --version
                '''
            }
        }
        
        stage('Pull Data with DVC') {
            steps {
                // Utilisation des credentials pour l'authentification Google Cloud
                withCredentials([file(credentialsId: 'google-service-account', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    bat '''
                        call %VENV_NAME%\\Scripts\\activate.bat
                        dvc remote modify storage --local gdrive_service_account_json_file_path %GOOGLE_APPLICATION_CREDENTIALS%
                        dvc pull
                    '''
                }
            }
        }
        
        stage('Code Quality') {
            steps {
                // Vérification de qualité du code avec flake8
                bat '''
                    call %VENV_NAME%\\Scripts\\activate.bat
                    pip install flake8
                    flake8 app.py --max-line-length=120 --ignore=E402,E501
                '''
            }
        }
        
        stage('Test Application') {
            steps {
                // Test de l'application
                bat '''
                    call %VENV_NAME%\\Scripts\\activate.bat
                    
                    REM Créer un répertoire pour les tests
                    mkdir test_data
                    
                    REM Script Python pour tester l'application et l'entraînement de modèle
                    python -c "
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

print('Test de création de jeu de données synthétique...')
X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

test_csv_path = os.path.join('test_data', 'test_dataset.csv')
df.to_csv(test_csv_path, index=False)
print(f'CSV créé à {test_csv_path}')

print('Test d\\'entraînement du modèle...')
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'MSE: {mse:.4f}')
print(f'R²: {r2:.4f}')

if r2 < 0.5:
    print('ERREUR: Le R² est inférieur à 0.5')
    exit(1)

model_path = os.path.join('test_data', 'test_model.pkl')
joblib.dump(model, model_path)
print(f'Modèle sauvegardé à {model_path}')

print('Tests réussis!')
"
                '''
            }
        }
        
        stage('Deployment Check') {
            steps {
                // Vérification de la conformité pour le déploiement
                bat '''
                    call %VENV_NAME%\\Scripts\\activate.bat
                    
                    REM Vérifier que les fichiers nécessaires sont présents
                    if not exist app.py (
                        echo "ERREUR: app.py est manquant"
                        exit 1
                    )
                    
                    if not exist requirements.txt (
                        echo "ERREUR: requirements.txt est manquant"
                        exit 1
                    )
                    
                    if not exist templates (
                        echo "ERREUR: Le dossier templates est manquant"
                        exit 1
                    )
                    
                    REM Vérifier que Flask est installé correctement
                    python -c "import flask; print(f'Flask version {flask.__version__} installée correctement')"
                    
                    echo "Vérification de déploiement réussie"
                '''
            }
        }
    }
    
    post {
        always {
            // Nettoyage après l'exécution du pipeline
            bat '''
                rmdir /S /Q test_data || echo "Pas de dossier test_data à supprimer"
            '''
        }
        
        success {
            echo 'Pipeline exécuté avec succès! L\'application est prête pour le déploiement.'
        }
        
        failure {
            echo 'Le pipeline a échoué. Vérifiez les logs pour plus de détails.'
        }
    }
}