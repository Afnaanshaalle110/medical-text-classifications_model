from flask import Flask, render_template, request, jsonify, url_for, redirect, flash, session
import requests
import json
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_pymongo import PyMongo
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import os
from functools import wraps
from bson.objectid import ObjectId # Import ObjectId
from datetime import datetime # Import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_secret_key') # Replace with a strong secret key

# Load environment variables from .env file
load_dotenv()

# MongoDB Configuration
try:
    # Try to get MongoDB URI from environment
    mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://medical:QhwA0wQYVOoTwiB2@cluster0.bqplbnl.mongodb.net/MTT_classification?retryWrites=true&w=majority")
    
    # If it's a MongoDB Atlas URI, add SSL configuration
    if "mongodb.net" in mongo_uri:
        # Add SSL configuration for Atlas
        if "?" not in mongo_uri:
            mongo_uri += "?ssl=true&tlsAllowInvalidCertificates=true"
        else:
            mongo_uri += "&ssl=true&tlsAllowInvalidCertificates=true"
    
    app.config["MONGO_URI"] = mongo_uri
    mongo = PyMongo(app)
    
    # Test the connection
    mongo.db.command('ping')
    print("✅ MongoDB connected successfully")
    
except Exception as e:
    print(f"⚠️ MongoDB Atlas connection failed: {e}")
    print("🔄 Falling back to local MongoDB...")
    
    # Fallback to local MongoDB
    app.config["MONGO_URI"] = "mongodb+srv://medical:QhwA0wQYVOoTwiB2@cluster0.bqplbnl.mongodb.net/MTT_classification?retryWrites=true&w=majority"
    mongo = PyMongo(app)
    
    try:
        # Test local connection
        mongo.db.command('ping')
        print("✅ MongoDB Atlas connected successfully")
    except Exception as local_error:
        print(f"❌ Local MongoDB also failed: {local_error}")
        print("💡 Please ensure MongoDB is running locally or check your Atlas connection")
        # Continue anyway - the app will show database errors when needed

# Add this after the MongoDB configuration section
def handle_db_error(f):
    """Decorator to handle database connection errors"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            if "ServerSelectionTimeoutError" in str(e) or "ConnectionError" in str(e):
                flash('Database connection error. Please try again later.', 'danger')
                return render_template('error.html', error="Database connection failed"), 503
            else:
                flash('An error occurred. Please try again.', 'danger')
                return render_template('error.html', error="Internal server error"), 500
    return decorated_function

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_data):
        self.user_data = user_data
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.roles = user_data.get('roles', []) # Ensure roles exist

    def get_id(self):
        return self.id

    def has_role(self, role):
        return role in self.roles

@login_manager.user_loader
def load_user(user_id):
    user_data = mongo.db.users.find_one({'_id': ObjectId(user_id)}) # Use ObjectId directly
    if user_data:
        return User(user_data)
    return None

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8000"

@app.route('/')
@app.route('/index.html')
def home():
    return render_template('index.html', current_user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_data = mongo.db.users.find_one({'username': username})

        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data)
            login_user(user)
            flash('Logged in successfully!', 'success')
            print(f"User '{username}' logged in successfully. Roles: {user.roles}") # Log successful login and roles
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')
            if user_data:
                print(f"Login failed for user '{username}': Incorrect password. User roles: {user_data.get('roles', 'N/A')}") # Log password mismatch
            else:
                print(f"Login failed: User '{username}' not found.") # Log user not found

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        existing_user = mongo.db.users.find_one({'username': username})
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
        else:
            hashed_password = generate_password_hash(password)
            new_user = {
                'username': username,
                'password': hashed_password,
                'roles': ['user'] # Default role is user
            }
            mongo.db.users.insert_one(new_user)
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Decorator for admin-only access
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.has_role('admin'):
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

# Example of an admin-only route
@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html', current_user=current_user)

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    users = list(mongo.db.users.find()) # Convert cursor to a list
    return render_template('admin_users.html', current_user=current_user, users=users)

@app.route('/admin/analytics')
@login_required
@admin_required
def admin_analytics():
    """Admin analytics page with real data from database"""
    try:
        # Get total users
        total_users = mongo.db.users.count_documents({})
        
        # Get total classifications
        total_classifications = mongo.db.classifications.count_documents({})
        
        # Get classifications by specialty (top 10)
        pipeline = [
            {"$group": {"_id": "$predicted_specialty", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        specialty_stats = list(mongo.db.classifications.aggregate(pipeline))
        
        # Get recent classifications (last 10)
        recent_classifications = list(mongo.db.classifications.find().sort("timestamp", -1).limit(10))
        
        # Get user activity (users with most classifications)
        user_activity = list(mongo.db.classifications.aggregate([
            {"$group": {"_id": "$user_id", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]))
        
        # Get classifications by date (last 7 days)
        from datetime import datetime, timedelta
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        daily_stats = list(mongo.db.classifications.aggregate([
            {"$match": {"timestamp": {"$gte": seven_days_ago}}},
            {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}, "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]))
        
        analytics_data = {
            'total_users': total_users,
            'total_classifications': total_classifications,
            'specialty_stats': specialty_stats,
            'recent_classifications': recent_classifications,
            'user_activity': user_activity,
            'daily_stats': daily_stats
        }
        
        return render_template('admin_analytics.html', current_user=current_user, analytics=analytics_data)
        
    except Exception as e:
        print(f"Error loading analytics: {e}")
        # Return empty analytics if there's an error
        analytics_data = {
            'total_users': 0,
            'total_classifications': 0,
            'specialty_stats': [],
            'recent_classifications': [],
            'user_activity': [],
            'daily_stats': []
        }
        return render_template('admin_analytics.html', current_user=current_user, analytics=analytics_data)

@app.route('/admin/users/edit/<user_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_user(user_id):
    user_data = mongo.db.users.find_one({'_id': ObjectId(user_id)})
    if not user_data:
        flash('User not found.', 'danger')
        return redirect(url_for('admin_users'))

    if request.method == 'POST':
        new_username = request.form.get('username')
        new_roles = request.form.getlist('roles') # Assuming multiple checkboxes for roles

        update_data = {}
        if new_username and new_username != user_data['username']:
            # Check if new username already exists
            if mongo.db.users.find_one({'username': new_username, '_id': {'$ne': ObjectId(user_id)}}):
                flash(f'Username \'{new_username}\' already exists.', 'danger')
                return redirect(url_for('admin_edit_user', user_id=user_id))
            update_data['username'] = new_username

        if new_roles is not None and sorted(new_roles) != sorted(user_data.get('roles', [])):
            update_data['roles'] = new_roles

        if update_data:
            mongo.db.users.update_one({'_id': ObjectId(user_id)}, {'$set': update_data})
            flash('User updated successfully!', 'success')
        else:
            flash('No changes were made.', 'info')
            
        return redirect(url_for('admin_users'))
    
    return render_template('admin_edit_user.html', current_user=current_user, user=user_data)

@app.route('/admin/users/delete/<user_id>', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(user_id):
    # Prevent deleting the current logged-in user if they are an admin trying to delete themselves
    if str(current_user.id) == user_id:
        flash('You cannot delete your own admin account.', 'danger')
        return redirect(url_for('admin_users'))

    result = mongo.db.users.delete_one({'_id': ObjectId(user_id)})
    if result.deleted_count == 1:
        flash('User deleted successfully.', 'success')
    else:
        flash('User not found.', 'danger')
    return redirect(url_for('admin_users'))

@app.route('/classifier')
@app.route('/classifier.html')
@login_required
def classifier():
    return render_template('classifier.html')

@app.route('/specialties')
@app.route('/specialties.html')
@login_required
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
@login_required
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
            prediction_data = response.json()
            
            # Store classification data in MongoDB
            if current_user.is_authenticated:
                classification_entry = {
                    'user_id': ObjectId(current_user.id),
                    'input_text': data.get('text'),
                    'predicted_specialty': prediction_data.get('specialty'), # Assuming 'specialty' is in FastAPI response
                    'timestamp': datetime.utcnow() # Import datetime if not already
                }
                mongo.db.classifications.insert_one(classification_entry)
                print(f"Classification saved for user {current_user.username}")

            return jsonify(prediction_data)
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
@login_required
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

# Install uvicorn for ASGI support
# pip install uvicorn

