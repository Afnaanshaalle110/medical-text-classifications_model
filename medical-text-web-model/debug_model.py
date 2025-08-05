#!/usr/bin/env python3
"""
Debug script to test model loading and identify TF-IDF vectorizer issues
"""

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def test_model_loading():
    """Test if models can be loaded properly"""
    try:
        print("Testing model loading...")
        
        # Load SVM model
        svm_model = joblib.load('svm_best_model.pkl')
        print("✅ SVM model loaded successfully!")
        print(f"SVM model type: {type(svm_model)}")
        
        # Load TF-IDF vectorizer
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        print("✅ TF-IDF vectorizer loaded successfully!")
        print(f"TF-IDF vectorizer type: {type(tfidf_vectorizer)}")
        
        # Check if TF-IDF is fitted
        if hasattr(tfidf_vectorizer, 'vocabulary_'):
            print(f"✅ TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
        else:
            print("❌ TF-IDF vectorizer is not fitted!")
            return False
        
        # Load feature selector
        try:
            feature_selector = joblib.load('feature_selector.pkl')
            print("✅ Feature selector loaded successfully!")
        except Exception as e:
            print(f"⚠️ Feature selector not found: {e}")
            feature_selector = None
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def test_prediction():
    """Test a simple prediction"""
    try:
        print("\nTesting prediction...")
        
        # Load models
        svm_model = joblib.load('svm_best_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        
        # Test text
        test_text = "The patient is a 27-year-old female presenting with abdominal pain"
        
        # Clean text (simplified)
        cleaned_text = test_text.lower()
        
        # Transform with TF-IDF
        print("Transforming text with TF-IDF...")
        text_tfidf = tfidf_vectorizer.transform([cleaned_text])
        print(f"TF-IDF shape: {text_tfidf.shape}")
        print(f"TF-IDF non-zero elements: {text_tfidf.nnz}")
        
        # Make prediction
        prediction = svm_model.predict(text_tfidf)
        print(f"Prediction: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return False

def check_tfidf_fitted():
    """Check if TF-IDF vectorizer is properly fitted"""
    try:
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        
        print("\nChecking TF-IDF vectorizer...")
        print(f"Type: {type(tfidf_vectorizer)}")
        print(f"Has vocabulary: {hasattr(tfidf_vectorizer, 'vocabulary_')}")
        print(f"Has idf_: {hasattr(tfidf_vectorizer, 'idf_')}")
        print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_) if hasattr(tfidf_vectorizer, 'vocabulary_') else 'N/A'}")
        
        # Test transform
        test_text = "test medical text"
        result = tfidf_vectorizer.transform([test_text])
        print(f"Transform successful: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ TF-IDF vectorizer error: {e}")
        return False

if __name__ == "__main__":
    print("=== Model Debug Script ===\n")
    
    # Test 1: Model loading
    model_loaded = test_model_loading()
    
    # Test 2: TF-IDF fitting check
    tfidf_ok = check_tfidf_fitted()
    
    # Test 3: Prediction
    if model_loaded and tfidf_ok:
        prediction_ok = test_prediction()
    else:
        prediction_ok = False
    
    print("\n=== Summary ===")
    print(f"Model loading: {'✅' if model_loaded else '❌'}")
    print(f"TF-IDF fitting: {'✅' if tfidf_ok else '❌'}")
    print(f"Prediction test: {'✅' if prediction_ok else '❌'}") 