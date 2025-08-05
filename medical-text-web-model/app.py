from flask import Flask, render_template, request, jsonify, url_for, redirect, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from datetime import datetime, timedelta
from functools import wraps
from dotenv import load_dotenv
import requests
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_secret_key')

# ✅ Fixed MongoDB URI with correct DB name
app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb+srv://medical:QhwA0wQYVOoTwiB2@cluster0.bqplbnl.mongodb.net/MTT_classification?retryWrites=true&w=majority")
mongo = PyMongo(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_data):
        self.user_data = user_data
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.roles = user_data.get('roles', [])

    def get_id(self):
        return self.id

    def has_role(self, role):
        return role in self.roles

@login_manager.user_loader
def load_user(user_id):
    user_data = mongo.db.users.find_one({'_id': ObjectId(user_id)})
    return User(user_data) if user_data else None

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
            login_user(User(user_data))
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if mongo.db.users.find_one({'username': username}):
            flash('Username already exists.', 'danger')
        else:
            mongo.db.users.insert_one({
                'username': username,
                'password': generate_password_hash(password),
                'roles': ['user']
            })
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.has_role('admin'):
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return wrapper

@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html', current_user=current_user)

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    users = list(mongo.db.users.find())
    return render_template('admin_users.html', current_user=current_user, users=users)

@app.route('/admin/users/edit/<user_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_user(user_id):
    user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('admin_users'))
    if request.method == 'POST':
        username = request.form.get('username')
        roles = request.form.getlist('roles')
        mongo.db.users.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {'username': username, 'roles': roles}}
        )
        flash('User updated successfully.', 'success')
        return redirect(url_for('admin_users'))
    return render_template('admin_edit_user.html', user=user, current_user=current_user)

@app.route('/admin/users/delete/<user_id>', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(user_id):
    mongo.db.users.delete_one({'_id': ObjectId(user_id)})
    flash('User deleted successfully.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/analytics')
@login_required
@admin_required
def admin_analytics():
    try:
        stats = {
            'total_users': mongo.db.users.count_documents({}),
            'total_classifications': mongo.db.classifications.count_documents({}),
            'specialty_stats': list(mongo.db.classifications.aggregate([
                {"$group": {"_id": "$predicted_specialty", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ])),
            'recent_classifications': list(mongo.db.classifications.find().sort("timestamp", -1).limit(10)),
            'user_activity': list(mongo.db.classifications.aggregate([
                {"$group": {"_id": "$user_id", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 5}
            ])),
            'daily_stats': list(mongo.db.classifications.aggregate([
                {"$match": {"timestamp": {"$gte": datetime.utcnow() - timedelta(days=7)}}},
                {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}, "count": {"$sum": 1}}},
                {"$sort": {"_id": 1}}
            ]))
        }
        return render_template('admin_analytics.html', analytics=stats, current_user=current_user)
    except Exception as e:
        print("Error loading analytics:", e)
        return render_template('admin_analytics.html', analytics={}, current_user=current_user)

@app.route('/classifier')
@login_required
def classifier():
    return render_template('classifier.html')

@app.route('/specialties')
@login_required
def specialties():
    return render_template('specialties.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/images')
def docimages():
    images = [
        {'filename': 'ai robort classified medical speacial with loptop.png', 'explanation': 'AI classifying specialties.'},
        {'filename': 'ai roport using microscope.png', 'explanation': 'Deep analysis like a microscope.'},
        {'filename': 'Flux_Dev_A_detailed_futuristic_3D_illustration_of_a_highly_rea_3.jpg', 'explanation': 'Futuristic AI medical scene.'},
        {'filename': 'using_using microscope.png', 'explanation': 'Collaboration between AI and doctors.'},
    ]
    return render_template('docimages.html', images=images)

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        response = requests.post(f"{FASTAPI_URL}/predict/", json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            mongo.db.classifications.insert_one({
                'user_id': ObjectId(current_user.id),
                'input_text': data['text'],
                'predicted_specialty': result.get('specialty'),
                'timestamp': datetime.utcnow()
            })
            return jsonify(result)
        return jsonify({'error': 'FastAPI error', 'detail': response.text}), response.status_code
    except Exception as e:
        return jsonify({'error': 'Internal error', 'detail': str(e)}), 500

@app.route('/api/specialties')
@login_required
def get_specialties():
    try:
        response = requests.get(f"{FASTAPI_URL}/specialties", timeout=10)
        return jsonify(response.json()) if response.status_code == 200 else jsonify({'error': 'FastAPI error'}), response.status_code
    except Exception as e:
        return jsonify({'error': 'Internal error', 'detail': str(e)}), 500

@app.route('/api/health')
def health_check():
    try:
        resp = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        return jsonify({'flask_status': 'healthy', 'fastapi_status': 'healthy' if resp.status_code == 200 else 'unhealthy'})
    except Exception as e:
        return jsonify({'flask_status': 'healthy', 'fastapi_status': 'unhealthy', 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5010)
