#!/usr/bin/env python3
"""
Extract the exact class labels from the trained SVM model
"""

import joblib

def extract_classes():
    """Extract and display the exact class labels from the model"""
    
    try:
        # Load the SVM model
        model = joblib.load('svm_best_model.pkl')
        classes = model.classes_
        
        print(f"Total classes: {len(classes)}")
        print("\nClass labels (with exact formatting):")
        print("CLASS_LABELS = [")
        for i, cls in enumerate(classes):
            print(f"    '{cls}',")
        print("]")
        
        print(f"\nClass mapping:")
        for i, cls in enumerate(classes):
            print(f"{i}: {repr(cls)}")
            
        return classes
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    extract_classes() 