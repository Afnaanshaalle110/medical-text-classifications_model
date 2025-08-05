#!/usr/bin/env python3
"""
Test script to verify Spacy installation and functionality
"""

def test_spacy():
    """Test if Spacy is working properly"""
    try:
        print("Testing Spacy installation...")
        
        # Try to import spacy
        import spacy
        print("✅ Spacy imported successfully!")
        
        # Try to load the English model
        nlp = spacy.load('en_core_web_sm')
        print("✅ English model loaded successfully!")
        
        # Test basic functionality
        test_text = "The patient is a 27-year-old female presenting with abdominal pain"
        doc = nlp(test_text)
        
        print(f"✅ Text processing successful!")
        print(f"  - Number of tokens: {len(doc)}")
        print(f"  - First few tokens: {[token.text for token in doc[:5]]}")
        
        # Test lemmatization
        lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 1]
        print(f"  - Lemmatized tokens: {lemmas[:10]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Spacy error: {e}")
        return False

if __name__ == "__main__":
    print("=== Spacy Test ===\n")
    
    success = test_spacy()
    
    if success:
        print("\n✅ Spacy is working correctly!")
        print("The medical text classification should now work with advanced text processing.")
    else:
        print("\n❌ Spacy is not working properly.")
        print("The application will fall back to basic text processing.") 