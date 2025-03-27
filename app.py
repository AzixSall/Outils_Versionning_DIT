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

if __name__ == '__main__':
    app.run(debug=True)