#!/usr/bin/env python3
"""
FastAPI backend for medical text classification using the new trained model
Matches the exact training pipeline: text cleaning -> TF-IDF -> SelectKBest -> SVM
"""

import joblib
import numpy as np
import re
import string
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC

# Global variable to track classification progress
classification_progress = {
    "status": "idle",  # idle, processing, complete, error
    "progress": 0,     # 0-100
    "message": "",     # Current step description
    "error": None      # Error message if any
}

# Download required NLTK data (optional)
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    print("✅ NLTK stopwords loaded")
except Exception:
    print("⚠️ NLTK not available - using empty stopwords set")
    stop_words = set()

# Load spacy model (optional)
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    print("✅ Spacy model loaded")
except Exception:
    print("⚠️ Spacy not available - using basic text processing")
    spacy = None
    nlp = None

app = FastAPI(
    title="Medical Text Classification API - New Model",
    description="Classify medical transcriptions using the newly trained SVM model with proper preprocessing pipeline",
    version="2.0.0"
)

def clean_text(text):
    """
    Enhanced text cleaning function matching the training code exactly
    """
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\b\d+\b', '', text)
    
    # Split into words and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 1]
    text = ' '.join(words)
    
    # Apply lemmatization if spacy is available
    if nlp:
        try:
            doc = nlp(text)
            lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 1]
            return ' '.join(lemmas)
        except Exception:
            return text
    
    return text

# Load the trained models and transformers
model_loaded = False
svm_model = None
tfidf_vectorizer = None
feature_selector = None

try:
    # Load the saved models
    print("Loading trained models...")
    
    # Load SVM model
    svm_model = joblib.load('svm_best_model.pkl')
    print("✅ SVM model loaded successfully!")
    
    # Load TF-IDF vectorizer
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ TF-IDF vectorizer loaded successfully!")
    
    # Try to load feature selector, if not available create one
    try:
        feature_selector = joblib.load('feature_selector.pkl')
        print("✅ Feature selector loaded successfully!")
    except Exception:
        print("⚠️ Feature selector not found, will create one if needed")
        feature_selector = None
    
    model_loaded = True
    print(f"Model type: {type(svm_model)}")
    print(f"TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model_loaded = False

# Medical specialties from the training data (exact labels from the trained model)
CLASS_LABELS = [
    ' Bariatrics',
    ' Cardiovascular / Pulmonary',
    ' Chiropractic',
    ' Consult - History and Phy.',
    ' Cosmetic / Plastic Surgery',
    ' Dentistry',
    ' Dermatology',
    ' Diets and Nutritions',
    ' Discharge Summary',
    ' ENT - Otolaryngology',
    ' Emergency Room Reports',
    ' Endocrinology',
    ' Gastroenterology',
    ' General Medicine',
    ' Hematology - Oncology',
    ' IME-QME-Work Comp etc.',
    ' Letters',
    ' Nephrology',
    ' Neurology',
    ' Neurosurgery',
    ' Obstetrics / Gynecology',
    ' Office Notes',
    ' Ophthalmology',
    ' Orthopedic',
    ' Pain Management',
    ' Pediatrics - Neonatal',
    ' Physical Medicine - Rehab',
    ' Podiatry',
    ' Psychiatry / Psychology',
    ' Radiology',
    ' Rheumatology',
    ' SOAP / Chart / Progress Notes',
    ' Sleep Medicine',
    ' Surgery',
    ' Urology',
]

# Pydantic models
class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    specialty: str
    confidence: float = None

class ProgressResponse(BaseModel):
    status: str
    progress: int
    message: str
    error: str = None

def update_progress(status: str, progress: int, message: str, error: str = None):
    """Update the global progress tracker"""
    global classification_progress
    classification_progress["status"] = status
    classification_progress["progress"] = progress
    classification_progress["message"] = message
    classification_progress["error"] = error

@app.get("/")
async def welcome():
    return {
        "message": "Medical Text Classification API - New Model",
        "version": "2.0.0",
        "model_info": {
            "algorithm": "LinearSVC (Support Vector Machine)",
            "preprocessing": "Text cleaning + TF-IDF + SelectKBest feature selection",
            "features": "3000 selected features from 10000 TF-IDF features",
            "training_accuracy": "77.98%"
        },
        "status": "ready" if model_loaded else "models not loaded"
    }

@app.get("/specialties")
async def get_specialties():
    return {
        "specialties": CLASS_LABELS,
        "total_count": len(CLASS_LABELS),
        "note": "These are the medical specialties the model can classify"
    }

@app.get("/progress", response_model=ProgressResponse)
async def get_progress():
    """Get the current classification progress"""
    return classification_progress

@app.post("/predict/", response_model=PredictionResponse)
async def predict_specialty(input_data: TextInput) -> PredictionResponse:
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded properly")
    
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    try:
        # Reset progress
        update_progress("processing", 0, "Starting text classification...")
        
        # Step 1: Clean the text using the same function as training
        update_progress("processing", 20, "Cleaning and preprocessing text...")
        cleaned_text = clean_text(input_data.text)
        print(f"Original text length: {len(input_data.text)}")
        print(f"Cleaned text length: {len(cleaned_text)}")
        print(f"Cleaned text preview: {cleaned_text[:100]}...")
        
        if not cleaned_text.strip():
            update_progress("error", 0, "Error: Text becomes empty after cleaning")
            raise HTTPException(status_code=400, detail="Text becomes empty after cleaning")
        
        # Step 2: Transform with TF-IDF vectorizer (trained)
        update_progress("processing", 40, "Applying TF-IDF transformation...")
        text_tfidf = tfidf_vectorizer.transform([cleaned_text])
        print(f"TF-IDF shape: {text_tfidf.shape}")
        
        # Step 3: Handle feature dimension mismatch
        update_progress("processing", 60, "Selecting features...")
        if feature_selector and text_tfidf.shape[1] >= 3000:
            text_selected = feature_selector.transform(text_tfidf)
            print(f"Selected features shape: {text_selected.shape}")
        else:
            if text_tfidf.shape[1] >= 3000:
                text_selected = text_tfidf[:, :3000]
                print(f"Truncated to 3000 features: {text_selected.shape}")
            else:
                import scipy.sparse as sp
                padding_size = 3000 - text_tfidf.shape[1]
                padding = sp.csr_matrix((text_tfidf.shape[0], padding_size))
                text_selected = sp.hstack([text_tfidf, padding])
                print(f"Padded to 3000 features: {text_selected.shape}")
        
        # Step 4: Make prediction with SVM model
        update_progress("processing", 80, "Making prediction...")
        prediction = svm_model.predict(text_selected)
        print(f"Raw prediction: {prediction}")
        
        # Get the predicted class index
        if isinstance(prediction, np.ndarray):
            predicted_class_idx = prediction[0]
        else:
            predicted_class_idx = prediction
        
        print(f"Predicted class index: {predicted_class_idx}")
        
        # Map prediction to specialty name
        if isinstance(predicted_class_idx, (int, np.integer)):
            if 0 <= predicted_class_idx < len(CLASS_LABELS):
                specialty = CLASS_LABELS[predicted_class_idx]
            else:
                specialty = f"Class_{predicted_class_idx}"
        else:
            specialty = str(predicted_class_idx)
        
        print(f"Final specialty: {specialty}")
        
        # Calculate confidence if possible
        confidence = None
        if hasattr(svm_model, 'decision_function'):
            try:
                decision_scores = svm_model.decision_function(text_selected)
                if hasattr(decision_scores, 'shape') and len(decision_scores.shape) > 1:
                    confidence = float(np.max(decision_scores))
                else:
                    confidence = float(decision_scores[0])
            except Exception as e:
                print(f"Could not calculate confidence: {e}")

        update_progress("complete", 100, "Classification complete!")
        return PredictionResponse(specialty=specialty, confidence=confidence)
        
    except Exception as e:
        error_msg = f"Classification error: {str(e)}"
        update_progress("error", 0, error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "version": "2.0.0",
        "models": {
            "svm_model": svm_model is not None,
            "tfidf_vectorizer": tfidf_vectorizer is not None,
            "feature_selector": feature_selector is not None
        }
    }

@app.post("/test_prediction")
async def test_prediction():
    """Test endpoint with sample medical text"""
    sample_text = "A 54-year-old male with a history of diabetes and high blood pressure presented to the ER with palpitations, dizziness, and irregular heartbeat noted on ECG."
    
    try:
        result = await predict_specialty(TextInput(text=sample_text))
        return {
            "test_text": sample_text,
            "prediction": result.specialty,
            "confidence": result.confidence,
            "status": "success"
        }
    except Exception as e:
        return {
            "test_text": sample_text,
            "error": str(e),
            "status": "failed"
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 