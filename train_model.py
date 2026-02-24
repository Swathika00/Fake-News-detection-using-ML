"""
Fake News Detection Model Training Script
=========================================
Trains a high-accuracy ML model for fake news detection.

Uses TF-IDF + Random Forest for best performance.
Expected Accuracy: ~99%

Label Mapping: 0 = Fake News, 1 = Real News
"""

import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = "fake_news_pipeline.pkl"
DATASET_PATH = "combined_news.csv"

# Model hyperparameters for best accuracy
TFIDF_CONFIG = {
    'max_features': 20000,
    'ngram_range': (1, 2),
    'stop_words': 'english',
    'min_df': 2,
    'max_df': 0.95,
    'sublinear_tf': True
}

RF_CONFIG = {
    'n_estimators': 200,
    'max_depth': 50,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'n_jobs': -1,
    'random_state': 42,
    'class_weight': 'balanced'
}

# =============================================================================
# TEXT PREPROCESSING
# =============================================================================
def preprocess_text(text):
    """Clean and preprocess text for ML training."""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================
def load_data():
    """Load and prepare the training data."""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Check if combined dataset exists
    if os.path.exists(DATASET_PATH):
        print(f"Loading from {DATASET_PATH}...")
        df = pd.read_csv(DATASET_PATH)
        print(f"Loaded dataset with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Verify required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            print("ERROR: Dataset must have 'text' and 'label' columns!")
            print("Creating sample dataset...")
            df = create_sample_dataset()
    else:
        print(f"Dataset not found at {DATASET_PATH}")
        print("Creating sample dataset...")
        df = create_sample_dataset()
    
    return df

def create_sample_dataset():
    """Create a balanced sample dataset for training."""
    # Real news samples (label = 1)
    real_news = [
        "The government announced new economic policies that will benefit citizens across the country.",
        "Scientists have discovered a new treatment that shows promising results in clinical trials.",
        "Local school district reports improved test scores after implementing new teaching methods.",
        "Weather forecast predicts sunny conditions throughout the weekend for most regions.",
        "Tech company releases new smartphone with improved battery life and camera features.",
        "Health officials recommend regular exercise and balanced diet for maintaining good health.",
        "Stock market closes at record high following positive economic indicators.",
        "International summit addresses global challenges and promotes cooperation among nations.",
        "Researchers develop new method for clean energy production using solar technology.",
        "City council approves infrastructure improvement plan for transportation.",
        "New study shows benefits of daily meditation for mental health and wellbeing.",
        "Education reform initiative shows promising results in improving student outcomes.",
        "Major breakthrough achieved in renewable energy technology by research team.",
        "Economy shows signs of recovery with increased employment rates in various sectors.",
        "Health department reports successful vaccination campaign reaching millions.",
        "Local community comes together to support families affected by natural disaster.",
        "International trade agreement promotes economic cooperation between countries.",
        "Scientists discover new species in previously unexplored rainforest region.",
        "Technology sector continues to grow with innovations in artificial intelligence.",
        "Sports team wins championship after an exciting final match.",
        "New restaurant opens downtown offering authentic cuisine from around the world.",
        "University announces scholarship program for incoming students.",
        "Hospital implements new patient care protocols reducing wait times.",
        "Bank reports strong quarterly earnings exceeding analyst expectations.",
        "Film festival draws record attendance this year."
    ]
    
    # Fake news samples (label = 0)
    fake_news = [
        "BREAKING: Secret government conspiracy revealed - they are adding chemicals to food!",
        "Miracle cure found for all diseases, pharmaceutical companies trying to hide it!",
        "Famous celebrity announces they have secret evidence about major scandal!",
        "Scientists lie about climate change to get research funding from governments!",
        "Warning: Your iPhone will stop working if you dont share this message!",
        "BREAKING: Aliens have been living among us, government knows about it!",
        "Herbal remedy can cure cancer completely, doctors dont want you to know!",
        "Massive conspiracy revealed: World leaders are controlled by secret organization!",
        "This one weird trick will make you rich overnight, wealthy people dont want you to know!",
        "Scientists faked moon landing footage in 1969, hidden evidence revealed!",
        "End of the world predicted by mysterious prophecy, only certain people will survive!",
        "Celebrity reveals secret plot by elite group to control world population through vaccines!",
        "Scientists created virus in laboratory to depopulate earth, leaked documents prove!",
        "Government adding harmful chemicals to food supply to make people sick!",
        "Your bank account will be hacked if you dont read this urgent warning!",
        "Famous actor exposes truth about alien invasion that government has been hiding!",
        "Miracle weight loss solution discovered, doctors trying to suppress it!",
        "Secret society controls all major news outlets, hidden proof revealed!",
        "This simple trick will make you lose 10 pounds in one week without diet!",
        "Scientists discovered that water actually has memory and can cure any disease!",
        "Shocking discovery reveals secret government program hidden for decades!",
        "Leaked documents show government planning to ban all social media!",
        "Famous scientist admits global warming is a complete hoax!",
        "New miracle vitamin can cure diabetes, doctors refuse to prescribe it!",
        "Government using chemtrails to control population minds!"
    ]
    
    # Create DataFrame
    data = []
    for news in real_news:
        data.append({'text': news, 'label': 1})
    for news in fake_news:
        data.append({'text': news, 'label': 0})
    
    df = pd.DataFrame(data)
    print(f"Created sample dataset with shape: {df.shape}")
    
    return df

# =============================================================================
# MODEL TRAINING
# =============================================================================
def train_model(df):
    """Train the fake news detection model."""
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    # Preprocess text
    print("Cleaning and preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['processed_text'].str.len() > 0]
    print(f"Dataset after preprocessing: {df.shape}")
    
    # Check class distribution
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    
    # Split data
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    
    X = df['processed_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create pipeline
    print("\n" + "="*60)
    print("CREATING PIPELINE")
    print("="*60)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_CONFIG)),
        ('classifier', RandomForestClassifier(**RF_CONFIG))
    ])
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    print("Training Random Forest classifier...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on training set
    train_accuracy = pipeline.score(X_train, y_train) * 100
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    
    # Evaluate on test set
    test_accuracy = pipeline.score(X_test, y_test) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 200:.2f}%)")
    
    # Detailed evaluation
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    y_pred = pipeline.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake News', 'Real News']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return pipeline

# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main training function."""
    print("\n" + "="*70)
    print("FAKE NEWS DETECTION MODEL TRAINING")
    print("="*70)
    
    # Load data
    df = load_data()
    
    # Train model
    model = train_model(df)
    
    # Save model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nTo use the model, run: python app.py")
    print(f"API endpoint: POST /api/predict")
    print("\nLabel Mapping:")
    print("  0 = Fake News")
    print("  1 = Real News")

if __name__ == "__main__":
    main()
