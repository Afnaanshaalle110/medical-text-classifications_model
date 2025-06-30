from flask import Flask, render_template, request, jsonify
import requests
import json

app = Flask(__name__)

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8000"

@app.route('/')
@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/classifier')
@app.route('/classifier.html')
def classifier():
    return render_template('classifier.html')

@app.route('/specialties')
@app.route('/specialties.html')
def specialties():
    return render_template('specialties.html')

@app.route('/about')
@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route("/debug")
def debug_classifier():
    """Debug page for classifier display issues"""
    with open('debug_classifier.html', 'r') as f:
        return f.read()

@app.route("/test-badge")
def test_badge():
    """Test page for specialty badge display"""
    with open('test_specialty_display.html', 'r') as f:
        return f.read()

@app.route('/images')
def docimages():
    """Page to display medical AI images and explanations."""
    # List of images and their explanations
    images = [
        {
            'filename': 'ai robort classified medical speacial with loptop.png',
            'explanation': 'This image represents the core function of our web model: using artificial intelligence to analyze and classify medical text into specialties. The robot symbolizes the automated, intelligent backend that processes user input and delivers accurate specialty predictions.'
        },
        {
            'filename': 'ai roport using microscope.png',
            'explanation': 'This image highlights the model’s ability to examine medical details closely, much like a microscope. It reflects the deep text analysis and feature extraction performed by our pipeline before classification.'
        },
        {
            'filename': 'Flux_Dev_A_detailed_futuristic_3D_illustration_of_a_highly_rea_3.jpg',
            'explanation': 'This futuristic scene illustrates the advanced technology behind our web model, combining machine learning and medical expertise to provide state-of-the-art classification for healthcare professionals.'
        },
        {
            'filename': 'using_using microscope.png',
            'explanation': 'This image shows the collaboration between human expertise and AI. Our web model is designed to assist medical professionals by providing fast, reliable specialty classification, supporting better decision-making.'
        },
    ]
    return render_template('docimages.html', images=images)

# API Routes to communicate with FastAPI backend
@app.route('/api/predict', methods=['POST'])
def predict():
    """Proxy route to FastAPI prediction endpoint"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        # Make request to FastAPI backend
        response = requests.post(
            f"{FASTAPI_URL}/predict/",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                'error': f'FastAPI error: {response.status_code}',
                'detail': response.text
            }), response.status_code
            
    except requests.exceptions.ConnectionError:
        return jsonify({
            'error': 'Cannot connect to ML backend',
            'detail': 'Make sure FastAPI server is running on port 8000'
        }), 503
    except requests.exceptions.Timeout:
        return jsonify({
            'error': 'Request timeout',
            'detail': 'The prediction took too long to complete'
        }), 504
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'detail': str(e)
        }), 500

@app.route('/api/specialties', methods=['GET'])
def get_specialties():
    """Proxy route to FastAPI specialties endpoint"""
    try:
        response = requests.get(f"{FASTAPI_URL}/specialties", timeout=10)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                'error': f'FastAPI error: {response.status_code}',
                'detail': response.text
            }), response.status_code
            
    except requests.exceptions.ConnectionError:
        return jsonify({
            'error': 'Cannot connect to ML backend',
            'detail': 'Make sure FastAPI server is running on port 8000'
        }), 503
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'detail': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check health of both Flask and FastAPI backends"""
    try:
        # Check FastAPI backend
        fastapi_response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        fastapi_healthy = fastapi_response.status_code == 200
        
        return jsonify({
            'flask_status': 'healthy',
            'fastapi_status': 'healthy' if fastapi_healthy else 'unhealthy',
            'fastapi_details': fastapi_response.json() if fastapi_healthy else 'Connection failed'
        })
        
    except Exception as e:
        return jsonify({
            'flask_status': 'healthy',
            'fastapi_status': 'unhealthy',
            'fastapi_details': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5010) 

   