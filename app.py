from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, classification_report
import pickle
import joblib
import json
from werkzeug.utils import secure_filename
import io
import base64
from datetime import datetime

app = Flask(__name__)
app.secret_key = "dit_secret_key"

# Configuration
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv', 'txt'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Aucun fichier sélectionné')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('Aucun fichier sélectionné')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            df = pd.read_csv(file_path)
            session['filepath'] = file_path
            session['filename'] = filename
            
            columns = df.columns.tolist()
            dtypes = df.dtypes.astype(str).tolist()
            preview = df.head(5).to_html(classes='table table-striped table-bordered')
            
            stats = df.describe().to_html(classes='table table-striped table-bordered')
            
            missing_values = df.isnull().sum().to_dict()
            
            return render_template('dataset.html', 
                                  filename=filename,
                                  columns=columns,
                                  dtypes=dtypes,
                                  preview=preview,
                                  stats=stats,
                                  missing_values=missing_values)
        except Exception as e:
            flash(f'Erreur lors de l\'analyse du fichier: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Type de fichier non autorisé')
    return redirect(url_for('index'))

@app.route('/train', methods=['GET', 'POST'])
def train():
    if 'filepath' not in session:
        flash('Veuillez d\'abord télécharger un fichier de données')
        return redirect(url_for('index'))
    
    if request.method == 'GET':
        df = pd.read_csv(session['filepath'])
        columns = df.columns.tolist()
        
        return render_template('train.html', columns=columns)
    
    elif request.method == 'POST':
        target_column = request.form.get('target_column')
        features = request.form.getlist('features')
        model_type = request.form.get('model_type')
        test_size = float(request.form.get('test_size', 0.2))
        
        if not target_column or not features:
            flash('Veuillez sélectionner une colonne cible et des caractéristiques')
            return redirect(url_for('train'))
        
        df = pd.read_csv(session['filepath'])
        
        X = df[features]
        y = df[target_column]
        
        X = X.fillna(X.mean())
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = None
        is_classification = False
        
        if model_type == 'linear_regression':
            model = LinearRegression()
        elif model_type == 'random_forest_regressor':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'random_forest_classifier':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            is_classification = True
        elif model_type == 'svm':
            model = SVC(probability=True, random_state=42)
            is_classification = True
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = {}
        plots = {}
        
        if is_classification:
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Prédictions')
            plt.ylabel('Valeurs réelles')
            plt.title('Matrice de confusion')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            confusion_matrix_plot = base64.b64encode(image_png).decode('utf-8')
            plots['confusion_matrix'] = confusion_matrix_plot
            
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics['classification_report'] = report
            
        else:  # Régression
            metrics['mean_squared_error'] = mean_squared_error(y_test, y_pred)
            metrics['r2_score'] = r2_score(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            plt.xlabel('Valeurs réelles')
            plt.ylabel('Prédictions')
            plt.title('Prédictions vs Valeurs réelles')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            predictions_plot = base64.b64encode(image_png).decode('utf-8')
            plots['predictions'] = predictions_plot
        
        if 'random_forest' in model_type:
            plt.figure(figsize=(10, 6))
            importances = pd.Series(model.feature_importances_, index=features)
            importances = importances.sort_values(ascending=False)
            importances.plot(kind='bar')
            plt.title('Importance des caractéristiques')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            importance_plot = base64.b64encode(image_png).decode('utf-8')
            plots['feature_importance'] = importance_plot
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_type}_{timestamp}.pkl"
        model_path = os.path.join(MODELS_FOLDER, model_filename)
        joblib.dump(model, model_path)
        
        metadata = {
            'model_type': model_type,
            'features': features,
            'target': target_column,
            'metrics': metrics,
            'timestamp': timestamp,
            'dataset_filename': session['filename']
        }
        
        metadata_filename = f"{model_type}_{timestamp}_metadata.json"
        metadata_path = os.path.join(MODELS_FOLDER, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        session['model_path'] = model_path
        session['model_type'] = model_type
        session['metrics'] = metrics
        session['plots'] = plots
        
        return render_template('results.html', 
                             model_type=model_type,
                             metrics=metrics,
                             plots=plots,
                             features=features,
                             target=target_column)
@app.route('/models')
def list_models():
    models = []
    
    for filename in os.listdir(MODELS_FOLDER):
        if filename.endswith('_metadata.json'):
            metadata_path = os.path.join(MODELS_FOLDER, filename)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                models.append(metadata)
    
    return render_template('models.html', models=models)

if __name__ == '__main__':
    app.run(debug=True)