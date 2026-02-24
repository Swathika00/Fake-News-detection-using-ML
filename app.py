"""
Fake News Detection Web Application
==================================
Flask web application with user authentication, 
single prediction, batch predictions, and history tracking.

Label Mapping: 0 = Fake News, 1 = Real News

Author: Fake News Detection Team
"""

import os
import re
import joblib
import pandas as pd
from datetime import datetime
from flask import (Flask, render_template, request, redirect, 
                  url_for, session, flash, jsonify)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.secret_key = os.environ.get('SECRET_KEY', 'fake-news-secret-key-2024')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = SQLAlchemy(app)

# Label mapping as per requirements: 0 = Fake, 1 = Real
LABEL_MAPPING = {
    0: "Fake News",
    1: "Real News"
}

# Model configuration
MODEL_PATH = "fake_news_pipeline.pkl"

# =============================================================================
# DATABASE MODELS
# =============================================================================
class User(db.Model):
    """User model for authentication."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with predictions
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    """Prediction history model."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.Integer, nullable=False)  # 0 = Fake, 1 = Real
    confidence = db.Column(db.Float, nullable=False)
    prediction_type = db.Column(db.String(20), default='single')  # single or batch
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# =============================================================================
# MODEL LOADING
# =============================================================================
def load_model():
    """Load the trained model pipeline."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# Load model at startup
model = load_model()

# =============================================================================
# TEXT PREPROCESSING
# =============================================================================
def preprocess_text(text):
    """Preprocess text for prediction."""
    if not text or pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def predict_news(text):
    """
    Make prediction on news text.
    Returns: (prediction, confidence, label)
    """
    if model is None:
        return None, None, "Model not loaded"
    
    processed_text = preprocess_text(text)
    if not processed_text:
        return None, None, "Empty text"
    
    try:
        prediction = model.predict([processed_text])[0]
        probabilities = model.predict_proba([processed_text])[0]
        confidence = float(probabilities[prediction] * 100)
        label = LABEL_MAPPING[prediction]
        return prediction, confidence, label
    except Exception as e:
        return None, None, str(e)

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Home page - redirects to login or dashboard."""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/home')
def home():
    """Home page with prediction form."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user=User.query.get(session['user_id']))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration."""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Validate input
        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return render_template('register.html')
        
        # Check if user exists
        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash('Username or email already exists!', 'error')
            return render_template('register.html')
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout."""
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    """User dashboard with prediction history."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    predictions = Prediction.query.filter_by(user_id=session['user_id'])\
        .order_by(Prediction.created_at.desc()).limit(50).all()
    
    return render_template('dashboard.html', 
                           user=user, 
                           predictions=predictions,
                           label_mapping=LABEL_MAPPING)

@app.route('/predict', methods=['POST'])
def predict():
    """Single news prediction."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    text = request.form.get('news_text', '').strip()
    
    if not text:
        flash('Please enter some news text to analyze.', 'error')
        return redirect(url_for('dashboard'))
    
    # Make prediction
    prediction, confidence, label = predict_news(text)
    
    if prediction is None:
        flash(f'Prediction error: {label}', 'error')
        return redirect(url_for('dashboard'))
    
    # Save to history
    new_prediction = Prediction(
        user_id=session['user_id'],
        text=text[:500],  # Limit text length for storage
        prediction=prediction,
        confidence=confidence,
        prediction_type='single'
    )
    db.session.add(new_prediction)
    db.session.commit()
    
    return render_template('result.html',
                           text=text[:200] + "..." if len(text) > 200 else text,
                           prediction=label,
                           confidence=confidence,
                           is_fake=prediction == 0)

@app.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    """Batch prediction from CSV file."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected!', 'error')
            return redirect(url_for('batch_predict'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected!', 'error')
            return redirect(url_for('batch_predict'))
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Read CSV file
                df = pd.read_csv(filepath)
                
                # Check if 'text' column exists
                if 'text' not in df.columns:
                    flash('CSV must have a "text" column!', 'error')
                    return redirect(url_for('batch_predict'))
                
                # Make predictions
                results = []
                for idx, row in df.iterrows():
                    text = str(row['text'])
                    prediction, confidence, label = predict_news(text)
                    
                    if prediction is not None:
                        # Save to history
                        new_pred = Prediction(
                            user_id=session['user_id'],
                            text=text[:500],
                            prediction=prediction,
                            confidence=confidence,
                            prediction_type='batch'
                        )
                        db.session.add(new_pred)
                        
                        results.append({
                            'text': text[:100] + "..." if len(text) > 100 else text,
                            'prediction': label,
                            'confidence': round(confidence, 2)
                        })
                
                db.session.commit()
                
                # Clean up uploaded file
                os.remove(filepath)
                
                return render_template('batch_result.html',
                                       results=results,
                                       total=len(results))
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(url_for('batch_predict'))
    
    return render_template('batch.html')

@app.route('/history')
def history():
    """View prediction history."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    predictions = Prediction.query.filter_by(user_id=session['user_id'])\
        .order_by(Prediction.created_at.desc()).all()
    
    return render_template('history.html',
                           predictions=predictions,
                           label_mapping=LABEL_MAPPING)

@app.route('/delete_history/<int:id>')
def delete_prediction(id):
    """Delete a single prediction from history."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    prediction = Prediction.query.filter_by(id=id, user_id=session['user_id']).first()
    
    if prediction:
        db.session.delete(prediction)
        db.session.commit()
        flash('Prediction deleted successfully.', 'success')
    
    return redirect(url_for('history'))

# =============================================================================
# API ENDPOINTS (for FastAPI integration)
# =============================================================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    prediction, confidence, label = predict_news(text)
    
    if prediction is None:
        return jsonify({'error': label}), 500
    
    # Convert to native Python types for JSON serialization
    return jsonify({
        'prediction': int(prediction),
        'confidence': float(round(confidence, 2)),
        'label': label
    })

@app.route('/api/health')
def api_health():
    """API health check."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Internal server error'), 500

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
    
    print("=" * 70)
    print("FAKE NEWS DETECTION WEB APPLICATION")
    print("=" * 70)
    print(f"Model status: {'Loaded' if model else 'Not available'}")
    print("\nStarting Flask server...")
    print("Open http://127.0.0.1:5000 in your browser")
    print("\nDefault routes:")
    print("  /          - Home (redirects to login)")
    print("  /register  - User registration")
    print("  /login     - User login")
    print("  /dashboard - Main dashboard with predictions")
    print("  /batch     - Batch predictions from CSV")
    print("  /history   - Prediction history")
    print("  /api/predict - API endpoint")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
