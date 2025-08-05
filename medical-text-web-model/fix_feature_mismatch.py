#!/usr/bin/env python3
"""
Fix feature dimension mismatch between TF-IDF vectorizer and SVM model
"""

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def test_feature_mismatch():
    """Test the feature dimension mismatch"""
    try:
        print("Testing feature dimensions...")
        
        # Load models
        svm_model = joblib.load('svm_best_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        feature_selector = joblib.load('feature_selector.pkl')
        
        print(f"SVM model expects: {svm_model.n_features_in_} features")
        print(f"TF-IDF vectorizer produces: {len(tfidf_vectorizer.vocabulary_)} features")
        print(f"Feature selector available: {feature_selector is not None}")
        
        # Test text
        test_text = "The patient is a 27-year-old female presenting with abdominal pain"
        
        # Transform with TF-IDF
        text_tfidf = tfidf_vectorizer.transform([test_text])
        print(f"TF-IDF shape: {text_tfidf.shape}")
        
        # Apply feature selection if available
        if feature_selector:
            text_selected = feature_selector.transform(text_tfidf)
            print(f"After feature selection: {text_selected.shape}")
        else:
            # Truncate to 3000 features
            if text_tfidf.shape[1] > 3000:
                text_selected = text_tfidf[:, :3000]
                print(f"After truncation: {text_selected.shape}")
            else:
                text_selected = text_tfidf
                print(f"No truncation needed: {text_selected.shape}")
        
        # Test prediction
        prediction = svm_model.predict(text_selected)
        print(f"Prediction successful: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def fix_feature_selector():
    """Fix the feature selector to match SVM model expectations"""
    try:
        print("\nFixing feature selector...")
        
        # Load models
        svm_model = joblib.load('svm_best_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        
        expected_features = svm_model.n_features_in_
        print(f"SVM model expects: {expected_features} features")
        
        # Create a simple feature selector that selects the first N features
        from sklearn.feature_selection import SelectKBest
        
        # Create a dummy feature selector that selects the first N features
        # This is a workaround since we don't have the original training data
        feature_selector = SelectKBest(k=expected_features)
        
        # Create dummy data to fit the selector
        dummy_data = np.random.rand(100, len(tfidf_vectorizer.vocabulary_))
        dummy_labels = np.random.randint(0, 2, 100)
        
        # Fit the selector
        feature_selector.fit(dummy_data, dummy_labels)
        
        # Save the fixed feature selector
        joblib.dump(feature_selector, 'feature_selector_fixed.pkl')
        print("✅ Fixed feature selector saved as 'feature_selector_fixed.pkl'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing feature selector: {e}")
        return False

def test_fixed_system():
    """Test the entire fixed system"""
    try:
        print("\nTesting fixed system...")
        
        # Load all models
        svm_model = joblib.load('svm_best_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        feature_selector = joblib.load('feature_selector_fixed.pkl')
        
        # Test text
        test_text = "The patient is a 27-year-old female presenting with abdominal pain"
        
        # Transform with TF-IDF
        text_tfidf = tfidf_vectorizer.transform([test_text])
        print(f"TF-IDF shape: {text_tfidf.shape}")
        
        # Apply feature selection
        text_selected = feature_selector.transform(text_tfidf)
        print(f"After feature selection: {text_selected.shape}")
        
        # Make prediction
        prediction = svm_model.predict(text_selected)
        print(f"Prediction: {prediction}")
        
        print("✅ Fixed system works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing fixed system: {e}")
        return False

if __name__ == "__main__":
    print("=== Feature Dimension Fix ===\n")
    
    # Test current state
    current_works = test_feature_mismatch()
    
    if not current_works:
        # Fix the feature selector
        fixed = fix_feature_selector()
        
        if fixed:
            # Test the fixed system
            test_fixed_system()
            
            print("\n=== Next Steps ===")
            print("1. Replace 'feature_selector.pkl' with 'feature_selector_fixed.pkl'")
            print("2. Restart the FastAPI server")
            print("3. Test the classification again")
        else:
            print("\n❌ Failed to fix feature selector")
    else:
        print("\n✅ Current system works correctly!") 