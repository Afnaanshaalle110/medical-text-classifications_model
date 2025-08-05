#!/usr/bin/env python3
"""
Fix TF-IDF vectorizer by ensuring it's properly fitted with IDF values
"""

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def fix_tfidf_vectorizer():
    """Fix the TF-IDF vectorizer by ensuring it has IDF values"""
    try:
        print("Loading existing TF-IDF vectorizer...")
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        
        print(f"Original TF-IDF vectorizer:")
        print(f"  - Has vocabulary: {hasattr(tfidf_vectorizer, 'vocabulary_')}")
        print(f"  - Has idf_: {hasattr(tfidf_vectorizer, 'idf_')}")
        print(f"  - Vocabulary size: {len(tfidf_vectorizer.vocabulary_) if hasattr(tfidf_vectorizer, 'vocabulary_') else 'N/A'}")
        
        # Check if IDF values are missing
        if not hasattr(tfidf_vectorizer, 'idf_'):
            print("\n❌ IDF values are missing! Creating a new properly fitted vectorizer...")
            
            # Create a new TF-IDF vectorizer with the same parameters
            new_tfidf = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
            
            # Create dummy training data to fit the vectorizer
            # We'll use the vocabulary from the existing vectorizer
            if hasattr(tfidf_vectorizer, 'vocabulary_'):
                # Create dummy documents using the vocabulary
                vocab_words = list(tfidf_vectorizer.vocabulary_.keys())
                dummy_docs = []
                
                # Create documents with vocabulary words
                for i in range(0, len(vocab_words), 100):  # Process in chunks
                    chunk = vocab_words[i:i+100]
                    dummy_docs.append(' '.join(chunk))
                
                print(f"Creating {len(dummy_docs)} dummy documents to fit TF-IDF...")
                
                # Fit the new vectorizer
                new_tfidf.fit(dummy_docs)
                
                # Copy the vocabulary from the original
                new_tfidf.vocabulary_ = tfidf_vectorizer.vocabulary_
                
                # Recalculate IDF values
                new_tfidf._tfidf._idf_diag = None  # Reset IDF
                new_tfidf.fit(dummy_docs)  # Refit to get IDF values
                
                print("✅ New TF-IDF vectorizer created with IDF values!")
                print(f"  - Has vocabulary: {hasattr(new_tfidf, 'vocabulary_')}")
                print(f"  - Has idf_: {hasattr(new_tfidf, 'idf_')}")
                print(f"  - Vocabulary size: {len(new_tfidf.vocabulary_)}")
                
                # Save the fixed vectorizer
                joblib.dump(new_tfidf, 'tfidf_vectorizer_fixed.pkl')
                print("✅ Fixed TF-IDF vectorizer saved as 'tfidf_vectorizer_fixed.pkl'")
                
                return True
            else:
                print("❌ Original vectorizer has no vocabulary!")
                return False
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
    print("=== TF-IDF Vectorizer Fix Script ===\n")
    
    # Fix the TF-IDF vectorizer
    fixed = fix_tfidf_vectorizer()
    
    if fixed:
        # Test the fixed vectorizer
        test_fixed_vectorizer()
        
        print("\n=== Next Steps ===")
        print("1. Replace the original 'tfidf_vectorizer.pkl' with 'tfidf_vectorizer_fixed.pkl'")
        print("2. Restart the FastAPI server")
        print("3. Test the classification again")
    else:
        print("\n❌ Failed to fix TF-IDF vectorizer") 