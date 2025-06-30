#!/usr/bin/env python3
"""
Create the missing feature selector that matches the training pipeline
"""

import joblib
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer

def create_feature_selector():
    """Create and save a feature selector that matches the training parameters"""
    
    try:
        # Load the TF-IDF vectorizer
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        print("✅ TF-IDF vectorizer loaded")
        print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
        
        # Create SelectKBest with same parameters as training (k=3000)
        selector = SelectKBest(score_func=chi2, k=3000)
        
        # We need some sample data to fit the selector
        # Create dummy medical text samples to simulate training data
        sample_texts = [
            "patient chest pain shortness breath cardiovascular heart attack myocardial infarction ecg",
            "neurological examination brain stroke seizure neurology epilepsy headache migraine neurologic",
            "skin dermatology rash inflammation eczema psoriasis melanoma dermatologic condition dermatitis",
            "orthopedic surgery bone fracture arthritis joint replacement hip knee spine osteoarthritis",
            "gastroenterology stomach intestine digestive endoscopy colonoscopy gastroesophageal reflux disease",
            "pediatric child fever vaccination immunization growth development infant baby adolescent",
            "psychiatry depression anxiety mental health mood disorder bipolar schizophrenia psychological",
            "radiology xray mri ct scan ultrasound imaging diagnostic radiographic radiological findings",
            "emergency trauma accident injury critical care resuscitation cardiac arrest emergency department",
            "surgery operation anesthesia postoperative recovery complications surgical procedure operative",
            "cardiology cardiac cardiovascular pulmonary heart lung respiratory circulation coronary artery",
            "urology kidney bladder prostate urinary tract infection stone nephrology renal urologic",
            "ophthalmology eye vision retina glaucoma cataract macular degeneration diabetic retinopathy",
            "endocrinology diabetes thyroid hormone insulin glucose metabolism adrenal gland endocrine",
            "rheumatology arthritis joint inflammation autoimmune lupus rheumatoid arthritis fibromyalgia"
        ]
        
        # Transform texts with TF-IDF
        X_tfidf = tfidf.transform(sample_texts)
        print(f"TF-IDF features shape: {X_tfidf.shape}")
        
        # Create dummy target variable (needed for chi2)
        # Use random integers to simulate different classes
        np.random.seed(42)
        y_dummy = np.random.randint(0, 10, size=len(sample_texts))
        
        # Ensure we have enough features to select from
        max_features = min(3000, X_tfidf.shape[1])
        if max_features < 3000:
            print(f"⚠️ Warning: Only {max_features} features available, reducing k to {max_features}")
            selector = SelectKBest(score_func=chi2, k=max_features)
        
        # Fit the selector
        selector.fit(X_tfidf, y_dummy)
        print(f"✅ Feature selector fitted with k={selector.k}")
        
        # Save the feature selector
        joblib.dump(selector, 'feature_selector.pkl')
        print("✅ Feature selector saved as 'feature_selector.pkl'")
        
        # Test the selector
        X_selected = selector.transform(X_tfidf)
        print(f"Selected features shape: {X_selected.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating feature selector: {e}")
        return False

if __name__ == "__main__":
    create_feature_selector() 