# MedClassify - Medical Specialty Classification Web App

MedClassify is a comprehensive web application that classifies medical transcriptions into different medical specialties using a rule-based approach.

## Features

- Simple web interface for entering medical transcription text
- Client-side classification of text into medical specialties
- Visualization of classification probabilities
- Sample texts for testing
- Detailed information about each medical specialty
- Multi-page website with responsive design

## Pages

1. **Home** (index.html) - Landing page with overview and features
2. **Classifier** (classifier.html) - The main classification tool
3. **Specialties** (specialties.html) - Detailed information about each medical specialty
4. **About** (about.html) - Project information, how it works, and contact form

## How to Use

Simply open the `index.html` file in your web browser. No server or installation required!

## How It Works

The application uses a rule-based approach to classify medical transcriptions:

1. It looks for keywords associated with different medical specialties in the input text
2. It counts the number of matches for each specialty
3. It calculates probability scores based on the match counts
4. It selects the specialty with the highest score as the prediction

## Supported Medical Specialties

The current implementation can classify text into the following specialties:
- Cardiology
- Orthopedics
- Neurology
- Pediatrics
- Dermatology
- General Surgery

## Technologies Used

- HTML5
- CSS3
- JavaScript
- Bootstrap 5

## Limitations

This is a demonstration application with the following limitations:

1. It uses a basic keyword-based approach rather than machine learning
2. The keyword list for each specialty is limited
3. It doesn't account for context or semantic meaning
4. All processing is done on the client side

## Future Development

Potential future enhancements include:
- Expanded keyword database
- Machine learning integration
- Natural language processing capabilities
- Additional medical specialties
- Server-side processing for advanced features
 





#install nltk and spacy

pip install nltk spacy
python -m nltk.downloader stopwords
python -m spacy download en_core_web_sm