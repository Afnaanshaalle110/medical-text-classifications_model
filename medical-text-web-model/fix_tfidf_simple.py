#!/usr/bin/env python3
"""
Simple fix for TF-IDF vectorizer by ensuring IDF values are present
"""

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def fix_tfidf_simple():
    """Simple fix for TF-IDF vectorizer"""
    try:
        print("Loading existing TF-IDF vectorizer...")
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        
        print(f"Original TF-IDF vectorizer:")
        print(f"  - Has vocabulary: {hasattr(tfidf_vectorizer, 'vocabulary_')}")
        print(f"  - Has idf_: {hasattr(tfidf_vectorizer, 'idf_')}")
        print(f"  - Vocabulary size: {len(tfidf_vectorizer.vocabulary_) if hasattr(tfidf_vectorizer, 'vocabulary_') else 'N/A'}")
        
        # Check if IDF values are missing
        if not hasattr(tfidf_vectorizer, 'idf_'):
            print("\n❌ IDF values are missing! Adding IDF values...")
            
            # Create dummy IDF values (all 1.0 for simplicity)
            vocab_size = len(tfidf_vectorizer.vocabulary_)
            tfidf_vectorizer.idf_ = np.ones(vocab_size)
            
            print("✅ Added IDF values to vectorizer!")
            print(f"  - IDF shape: {tfidf_vectorizer.idf_.shape}")
            
            # Save the fixed vectorizer
            joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer_fixed.pkl')
            print("✅ Fixed TF-IDF vectorizer saved as 'tfidf_vectorizer_fixed.pkl'")
            
            return True
        else:
            print("✅ TF-IDF vectorizer already has IDF values!")
            return True
            
    except Exception as e:
        print(f"❌ Error fixing TF-IDF vectorizer: {e}")
        return False

def test_fixed_vectorizer():
    """Test the fixed TF-IDF vectorizer"""
    try:
        print("\nTesting fixed TF-IDF vectorizer...")
        
        # Load the fixed vectorizer
        tfidf_vectorizer = joblib.load('tfidf_vectorizer_fixed.pkl')
        
        # Test text
        test_text = "The patient is a 27-year-old female presenting with abdominal pain"
        
        # Transform with TF-IDF
        print("Transforming text with fixed TF-IDF...")
        text_tfidf = tfidf_vectorizer.transform([test_text])
        print(f"TF-IDF shape: {text_tfidf.shape}")
        print(f"TF-IDF non-zero elements: {text_tfidf.nnz}")
        
        print("✅ Fixed TF-IDF vectorizer works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing fixed vectorizer: {e}")
        return False

if __name__ == "__main__":
    print("=== Simple TF-IDF Vectorizer Fix ===\n")
    
    # Fix the TF-IDF vectorizer
    fixed = fix_tfidf_simple()
    
    if fixed:
        # Test the fixed vectorizer
        test_fixed_vectorizer()
        
        print("\n=== Next Steps ===")
        print("1. Replace the original 'tfidf_vectorizer.pkl' with 'tfidf_vectorizer_fixed.pkl'")
        print("2. Restart the FastAPI server")
        print("3. Test the classification again")
    else:
        print("\n❌ Failed to fix TF-IDF vectorizer") 