# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_required, current_user, login_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import os
import tempfile
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Import the model after initializing Flask app
from model import preprocess_image, predict_image, model_path

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    role = db.Column(db.String(20), default='user')  # 'user' or 'admin'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)
    
    def is_admin(self):
        return self.role == 'admin'

# Prediction Model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False)
    patient_age = db.Column(db.Integer, nullable=False)
    patient_gender = db.Column(db.String(10))
    predicted_class = db.Column(db.Integer, nullable=False)
    prediction_label = db.Column(db.String(20))
    confidence_score = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    notes = db.Column(db.Text)
    image_path = db.Column(db.String(255))
    
# Class mappings
CLASS_MAPPINGS = {
    0: "Cancer",
    1: "Normal"
}

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Admin required decorator
def admin_required(f):
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin():
            flash("Admin access required.")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
            
        if User.query.filter_by(email=email).first():
            flash('Email already in use')
            return redirect(url_for('register'))
            
        user = User(username=username, email=email)
        user.set_password(password)
        
        # Make the first user an admin
        if User.query.count() == 0:
            user.role = 'admin'
            
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's predictions with pagination
    page = request.args.get('page', 1, type=int)
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).paginate(page=page, per_page=10)
    
    # Get statistics
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    cancer_count = Prediction.query.filter_by(user_id=current_user.id, prediction_label='Cancer').count()
    normal_count = Prediction.query.filter_by(user_id=current_user.id, prediction_label='Normal').count()
    
    # Create statistics chart
    if total_predictions > 0:
        labels = ['Cancer', 'Normal']
        counts = [cancer_count, normal_count]
        
        plt.figure(figsize=(6, 4))
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#66B2FF'])
        plt.axis('equal')
        plt.title('Diagnosis Distribution')
        
        # Save chart to a bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        chart_img = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
    else:
        chart_img = None
    
    return render_template('dashboard.html', 
                           predictions=predictions, 
                           total_predictions=total_predictions,
                           cancer_count=cancer_count,
                           normal_count=normal_count,
                           chart_img=chart_img)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image uploaded')
            return redirect(request.url)
            
        image = request.files['image']
        if image.filename == '':
            flash('No image selected')
            return redirect(request.url)
            
        # Get form data
        patient_name = request.form['patient_name']
        patient_age = request.form['patient_age']
        patient_gender = request.form.get('patient_gender', '')
        
        # Save the image to a temporary file
        img_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{image.filename}"
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        image.save(img_path)
        
        # Make prediction
        try:
            predictions = predict_image(img_path, model_path)
            predicted_class = np.argmax(predictions)
            confidence_score = float(predictions[0][predicted_class])
            prediction_label = CLASS_MAPPINGS.get(predicted_class, "Unknown")
            
            # Create prediction record
            prediction = Prediction(
                patient_name=patient_name,
                patient_age=patient_age,
                patient_gender=patient_gender,
                predicted_class=predicted_class,
                prediction_label=prediction_label,
                confidence_score=confidence_score,
                user_id=current_user.id,
                image_path=img_path
            )
            db.session.add(prediction)
            db.session.commit()

            return redirect(url_for('prediction_result', prediction_id=prediction.id))
        
        except Exception as e:
            flash(f'Error during prediction: {str(e)}')
            # Delete image if prediction fails
            if os.path.exists(img_path):
                os.remove(img_path)
            return redirect(url_for('predict'))
    
    return render_template('predict.html')

@app.route('/prediction/<int:prediction_id>')
@login_required
def prediction_result(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    if prediction.user_id != current_user.id and not current_user.is_admin():
        flash('Access denied')
        return redirect(url_for('dashboard'))
    
    return render_template('results.html', prediction=prediction)

@app.route('/api/predictions')
@login_required
def get_predictions():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return jsonify([{
        'id': p.id,
        'patient_name': p.patient_name,
        'patient_age': p.patient_age,
        'patient_gender': p.patient_gender,
        'predicted_class': p.predicted_class,
        'prediction_label': p.prediction_label,
        'confidence_score': p.confidence_score,
        'timestamp': p.timestamp.isoformat(),
        'notes': p.notes,
        'image_path': p.image_path
    } for p in predictions])

@app.route('/api/notes/<int:prediction_id>', methods=['POST'])
@login_required
def update_notes(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    if prediction.user_id != current_user.id and not current_user.is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    prediction.notes = data.get('notes', '')
    db.session.commit()
    return jsonify({'success': True})

@app.route('/admin')
@admin_required
def admin_dashboard():
    # Admin statistics
    total_users = User.query.count()
    total_predictions = Prediction.query.count()
    recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
    recent_predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(10).all()
    
    # Get cancer vs normal stats
    cancer_count = Prediction.query.filter_by(prediction_label='Cancer').count()
    normal_count = Prediction.query.filter_by(prediction_label='Normal').count()
    
    return render_template('admin.html', 
                           total_users=total_users,
                           total_predictions=total_predictions,
                           recent_users=recent_users,
                           recent_predictions=recent_predictions,
                           cancer_count=cancer_count,
                           normal_count=normal_count)

@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    
    # Check if user has permission to delete
    if prediction.user_id != current_user.id and not current_user.is_admin():
        flash('You do not have permission to delete this prediction')
        return redirect(url_for('dashboard'))
    
    # Delete the image file if it exists
    if prediction.image_path and os.path.exists(prediction.image_path):
        os.remove(prediction.image_path)
    
    # Delete the prediction from the database
    db.session.delete(prediction)
    db.session.commit()
    
    flash('Prediction deleted successfully')
    return redirect(url_for('dashboard'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        # Update email
        email = request.form.get('email')
        if email and email != current_user.email:
            if User.query.filter_by(email=email).first():
                flash('Email already in use')
            else:
                current_user.email = email
        
        # Change password if provided
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if current_password and new_password and confirm_password:
            if not current_user.check_password(current_password):
                flash('Current password is incorrect')
            elif new_password != confirm_password:
                flash('New passwords do not match')
            else:
                current_user.set_password(new_password)
                flash('Password updated successfully')
        
        db.session.commit()
        return redirect(url_for('profile'))
    
    return render_template('profile.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)