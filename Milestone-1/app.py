import os
import json
import time
import traceback
import csv
import uuid
import re
from datetime import datetime, timedelta, timezone
from io import StringIO
from functools import wraps
import shutil
import base64

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, session, \
    send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import func, desc, text, inspect
import flask_sqlalchemy
from dotenv import load_dotenv

load_dotenv()

# Create Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Configuration
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "super_secret_key_123")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ECHO"] = False  # Set to False to reduce logs

# Database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
database_url = os.getenv('DATABASE_URL')

if database_url:
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    print(f"📁 Using PostgreSQL database")
else:
    # Use a FIXED, permanent path for SQLite database
    db_dir = os.path.join(basedir, "data")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "agrobot.db")
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    print(f"📁 SQLite Database file: {db_path}")

# File upload configuration
UPLOAD_FOLDER = os.path.join(basedir, "static", "uploads")
THUMBNAIL_FOLDER = os.path.join(basedir, "static", "thumbnails")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["THUMBNAIL_FOLDER"] = THUMBNAIL_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_DOC_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx', 'csv'}

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Get Gemini API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_ENABLED = False
gemini_client = None

# Try to import and configure Gemini
try:
    # Try new package first
    try:
        from google import genai as google_genai

        if GEMINI_API_KEY and GEMINI_API_KEY != "not_set":
            gemini_client = google_genai.Client(api_key=GEMINI_API_KEY)
            GEMINI_ENABLED = True
            print(f"🔑 Gemini configured (new package)")
    except ImportError:
        # Try old package
        try:
            import google.generativeai as genai_old

            if GEMINI_API_KEY and GEMINI_API_KEY != "not_set":
                genai_old.configure(api_key=GEMINI_API_KEY)
                gemini_client = genai_old
                GEMINI_ENABLED = True
                print(f"🔑 Gemini configured (old package)")
        except ImportError:
            print("⚠️ No Gemini package found")
except Exception as e:
    print(f"⚠️ Gemini configuration failed: {e}")
    GEMINI_ENABLED = False

if not GEMINI_API_KEY or GEMINI_API_KEY == "not_set":
    print("⚠️ Warning: GEMINI_API_KEY not found in .env file")


# Helper function for UTC timestamps
def utc_now():
    return datetime.now(timezone.utc)


# ==================== DATABASE MODELS ====================

class User(db.Model, UserMixin):
    """User model for farmers and admins"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    whatsapp = db.Column(db.String(20))
    dob = db.Column(db.Date)
    gender = db.Column(db.String(20))
    profile_picture = db.Column(db.String(200))
    farm_name = db.Column(db.String(100))
    farm_size = db.Column(db.String(100), nullable=False, default='1-5')
    primary_crop = db.Column(db.String(100), nullable=False, default='Rice')
    secondary_crops = db.Column(db.Text)
    soil_type = db.Column(db.String(50))
    irrigation_type = db.Column(db.String(50))
    region = db.Column(db.String(100), nullable=False, default='')
    district = db.Column(db.String(100))
    farm_address = db.Column(db.Text)
    experience_level = db.Column(db.String(50), nullable=False, default='beginner')
    preferred_language = db.Column(db.String(10), nullable=False, default='en')

    # Notification preferences
    notify_weather = db.Column(db.Boolean, default=True)
    notify_pests = db.Column(db.Boolean, default=True)
    notify_market = db.Column(db.Boolean, default=True)
    notify_tips = db.Column(db.Boolean, default=True)

    # Interests
    interest_organic = db.Column(db.Boolean, default=False)
    interest_hydroponics = db.Column(db.Boolean, default=False)
    interest_precision = db.Column(db.Boolean, default=False)
    interest_dairy = db.Column(db.Boolean, default=False)
    interest_poultry = db.Column(db.Boolean, default=False)
    interest_fisheries = db.Column(db.Boolean, default=False)

    # Consent
    newsletter = db.Column(db.Boolean, default=True)
    share_data = db.Column(db.Boolean, default=False)

    # Referral
    referral_code = db.Column(db.String(50))
    points_balance = db.Column(db.Integer, default=0)

    # Account status
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    role = db.Column(db.String(20), default='farmer')

    # Timestamps - FIXED: Using lambda functions for proper default
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))
    last_login = db.Column(db.DateTime)
    email_verified_at = db.Column(db.DateTime)
    phone_verified_at = db.Column(db.DateTime)

    # Relationships
    chats = db.relationship('ChatHistory', backref='user', lazy=True, cascade='all, delete-orphan')
    image_analyses = db.relationship('ImageAnalysis', backref='user', lazy=True, cascade='all, delete-orphan')
    activities = db.relationship('UserActivity', backref='user', lazy=True, cascade='all, delete-orphan')
    points = db.relationship('UserPoints', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        """Hash and set password"""
        self.password = generate_password_hash(password)

    def check_password(self, password):
        """Check password hash"""
        return check_password_hash(self.password, password)

    def get_id(self):
        """Required by Flask-Login"""
        return str(self.id)

    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'phone': self.phone,
            'role': self.role,
            'region': self.region,
            'primary_crop': self.primary_crop,
            'farm_size': self.farm_size,
            'experience_level': self.experience_level,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ChatHistory(db.Model):
    """Chat history model"""
    __tablename__ = 'chat_history'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    bot_response = db.Column(db.Text, nullable=False)
    chat_type = db.Column(db.String(20), default='text')
    image_filename = db.Column(db.String(200))
    language = db.Column(db.String(10), default='en')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class ImageAnalysis(db.Model):
    """Image analysis model"""
    __tablename__ = 'image_analyses'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    thumbnail = db.Column(db.String(200))
    user_message = db.Column(db.Text)
    health_status = db.Column(db.String(100))
    analysis_result = db.Column(db.Text)
    confidence_score = db.Column(db.Float, default=0.0)
    crop_type = db.Column(db.String(100))
    severity_level = db.Column(db.String(50))
    recommendations = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class FarmingTip(db.Model):
    """Farming tips model"""
    __tablename__ = 'farming_tips'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), default='general')
    crop_type = db.Column(db.String(100))
    region = db.Column(db.String(100))
    language = db.Column(db.String(10), default='en')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class MarketPrice(db.Model):
    """Market prices model"""
    __tablename__ = 'market_prices'

    id = db.Column(db.Integer, primary_key=True)
    crop_name = db.Column(db.String(100), nullable=False)
    market_name = db.Column(db.String(100), nullable=False)
    region = db.Column(db.String(100))
    price = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(20), default='kg')
    date = db.Column(db.Date, nullable=False)
    source = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class UserActivity(db.Model):
    """User activity log"""
    __tablename__ = 'user_activities'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    activity_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class UserPoints(db.Model):
    """User points system"""
    __tablename__ = 'user_points'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    points = db.Column(db.Integer, nullable=False)
    balance_after = db.Column(db.Integer, nullable=False)
    transaction_type = db.Column(db.String(20), nullable=False)
    reason = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class Referral(db.Model):
    """Referral system"""
    __tablename__ = 'referrals'

    id = db.Column(db.Integer, primary_key=True)
    referrer_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    referred_email = db.Column(db.String(120), nullable=False)
    referral_code = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), default='pending')
    points_awarded = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    referred_user_id = db.Column(db.Integer, db.ForeignKey('users.id'))


class WeatherAlert(db.Model):
    """Weather alerts"""
    __tablename__ = 'weather_alerts'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)
    severity = db.Column(db.String(20), nullable=False)
    message = db.Column(db.Text, nullable=False)
    region = db.Column(db.String(100))
    date = db.Column(db.Date, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class OTPVerification(db.Model):
    """OTP verification"""
    __tablename__ = 'otp_verifications'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    email = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    otp = db.Column(db.String(10), nullable=False)
    otp_type = db.Column(db.String(20), nullable=False)
    is_used = db.Column(db.Boolean, default=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


# ==================== HELPER FUNCTIONS ====================

def clean_phone_number(phone):
    """Clean phone number by removing non-digits"""
    if not phone:
        return None
    digits = re.sub(r'\D', '', phone)
    if len(digits) >= 10:
        return f"+{digits}" if not digits.startswith('+') else digits
    return None


def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


def generate_thumbnail(image_path, thumb_path, size=(200, 200)):
    """Generate thumbnail for image"""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img.thumbnail(size)
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            img.save(thumb_path, "JPEG")
        return True
    except Exception as e:
        print(f"Thumbnail generation error: {e}")
        return False


# Local Knowledge Base
LOCAL_KNOWLEDGE = {
    "rice": {
        "planting": "Plant rice during monsoon season (June-July) for Kharif crop.",
        "fertilizer": "Use NPK 10:26:26 at planting and urea at 30 days.",
        "water": "Maintain 2-3 inches of water in fields during vegetative stage.",
        "harvest": "Harvest when grains are hard and moisture is below 14%.",
    },
    "wheat": {
        "planting": "Plant wheat in October-November (Rabi season).",
        "fertilizer": "Apply NPK 12:32:16 at sowing (50kg/acre).",
        "harvest": "Harvest when grains are hard and moisture is below 14%.",
    },
    "maize": {
        "planting": "Plant maize in June-July for Kharif and January-February for Rabi.",
        "fertilizer": "Apply 60:40:20 kg NPK per acre at sowing.",
    },
    "pest": {
        "aphids": "Use neem oil spray (2ml per liter) weekly to control aphids.",
        "fungus": "Apply copper-based fungicide at first signs of fungal infection.",
    },
    "soil": {
        "test": "Get soil tested every 2-3 years for optimal fertilizer use.",
        "ph": "Most crops prefer soil pH between 6.0 and 7.0.",
    }
}


def get_local_response(user_input, user_profile):
    """Get response from local knowledge base"""
    try:
        user_input_lower = user_input.lower()

        for topic, details in LOCAL_KNOWLEDGE.items():
            if topic in user_input_lower:
                for subtopic, response in details.items():
                    if subtopic in user_input_lower:
                        return f"**{topic.capitalize()} {subtopic.capitalize()}:**\n{response}\n\n*Source: Local Agricultural Knowledge Base*"

                first_key = next(iter(details))
                return f"**{topic.capitalize()} Information:**\n{details[first_key]}\n\nFor more specific advice, please ask about planting, fertilizer, water, or harvest."

        # Common questions
        if any(word in user_input_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm AgroBot, your agricultural assistant. How can I help you with farming today?"

        if any(word in user_input_lower for word in ["help", "what can you do"]):
            return "I can help with:\n• Crop cultivation advice\n• Pest and disease identification\n• Soil and fertilizer recommendations\n• Irrigation guidance\n• Market price information\n• Weather impacts\n• Harvesting techniques\n\nJust ask me anything about farming!"

        return None

    except Exception as e:
        print(f"Error in get_local_response: {e}")
        return None


def get_enhanced_fallback_response(user_input, user_profile):
    """Get enhanced fallback response"""
    fallback_responses = [
        f"I understand you're asking about '{user_input}'. While I don't have specific information on this, I recommend:\n1. Consulting local agricultural extension officers\n2. Visiting your nearest Krishi Vigyan Kendra\n3. Checking with experienced farmers in your area",
        f"Regarding '{user_input}', this is a specialized topic. For accurate advice, please:\n• Contact your state agriculture department\n• Use the Kisan Call Center (Dial 1551)\n• Download the Kisan Suvidha mobile app",
        f"Thank you for your question about '{user_input}'. For detailed guidance, I suggest:\n1. Soil testing for precise fertilizer recommendations\n2. Weather-based crop planning\n3. Integrated Pest Management (IPM) practices"
    ]

    if user_profile.get('region'):
        region = user_profile['region']
        personalized = f"\n\nSince you're in {region}, consider contacting the {region} Agricultural University for region-specific advice."
        return fallback_responses[0] + personalized

    return fallback_responses[len(user_input) % len(fallback_responses)]


def analyze_with_gemini(image_path, user_message=""):
    """Analyze image with Gemini Vision with fallback"""
    if not GEMINI_ENABLED or not gemini_client:
        return fallback_image_analysis(image_path)

    try:
        # Read image as base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Determine MIME type
        if image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
            mime_type = 'image/jpeg'
        elif image_path.lower().endswith('.gif'):
            mime_type = 'image/gif'
        elif image_path.lower().endswith('.bmp'):
            mime_type = 'image/bmp'
        elif image_path.lower().endswith('.webp'):
            mime_type = 'image/webp'
        else:
            mime_type = 'image/jpeg'

        # Create prompt
        if user_message:
            prompt = f"Analyze this agricultural image. User query: {user_message}\n\nProvide:\n1. Plant/crop identification\n2. Health assessment\n3. Pest/disease detection if any\n4. Recommendations"
        else:
            prompt = "Analyze this agricultural image. Provide:\n1. Plant/crop identification\n2. Health assessment\n3. Pest/disease detection if any\n4. Recommendations"

        # Create content parts
        parts = [
            {"text": prompt},
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_b64
                }
            }
        ]

        # CORRECT VISION MODEL NAMES for new package
        models_to_try = [
            'gemini-2.5-flash'  # Basic vision model
        ]

        for model_name in models_to_try:
            try:
                print(f"  🤖 Trying vision model: {model_name}")
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=parts
                )

                if response and response.text:
                    print(f"  ✅ Vision analysis successful with model: {model_name}")
                    return {
                        "analysis": response.text,
                        "confidence": 0.85,
                        "health_status": "Analyzed with AI",
                        "recommendations": "Follow the analysis above",
                        "source": f"Gemini ({model_name})"
                    }
            except Exception as e:
                error_msg = str(e)
                print(f"  ✗ Vision model {model_name} error: {error_msg[:100]}")

                # If rate limited, use fallback immediately
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print("  ⚠️ Rate limited, using fallback analysis")
                    return fallback_image_analysis(image_path)
                if "404" in error_msg or "NOT_FOUND" in error_msg:
                    print(f"  ⚠️ Model {model_name} not found, trying next...")
                    continue
                continue

    except Exception as e:
        print(f"Gemini Vision error: {e}")

    # Fallback if all Gemini attempts fail
    return fallback_image_analysis(image_path)

def fallback_image_analysis(image_path):
    """Fallback analysis when Gemini fails"""
    try:
        from PIL import Image
        im = Image.open(image_path).convert('RGB').resize((200, 200))
        pixels = list(im.getdata())
        greens = sum(1 for r, g, b in pixels if g > r + 10 and g > b + 10)
        total = len(pixels)
        healthy_ratio = greens / total if total > 0 else 0

        if healthy_ratio < 0.05:
            status = "Severe discoloration / possible disease"
            advice = "Image shows low green content. Inspect plants for diseases or nutrient deficiency."
        elif healthy_ratio < 0.4:
            status = "Partial damage / early symptoms"
            advice = "Signs of stress detected. Check for pests, water stress, or nutrient issues."
        else:
            status = "Likely healthy leaf"
            advice = "Leaf appears healthy with good green coverage."

        return {
            "health_status": status,
            "analysis": advice,
            "green_percentage": round(healthy_ratio * 100, 1),
            "recommendations": [
                "Take multiple photos from different angles",
                "Check soil moisture levels",
                "Look for signs of pests on underside of leaves"
            ]
        }
    except Exception as img_error:
        return {
            "health_status": "Analysis failed",
            "analysis": "Unable to analyze image. Please try with a clearer photo.",
            "recommendations": ["Ensure good lighting", "Take photo against plain background"]
        }


def ask_gemini(user_input, user_profile=None):
    """Ask Gemini AI for response with fallback"""
    if not GEMINI_ENABLED or not gemini_client:
        return get_enhanced_fallback_response(user_input, user_profile)

    context = ""
    if user_profile:
        context = f"""
        User Profile:
        - Name: {user_profile.get('name', 'User')}
        - Region: {user_profile.get('region', 'Not specified')}
        - Primary Crop: {user_profile.get('primary_crop', 'Not specified')}
        - Experience: {user_profile.get('experience_level', 'Not specified')}
        """

    prompt = f"""You are AgroBot, an AI agricultural assistant for Indian farmers.

    {context}

    Guidelines:
    1. Provide practical, actionable advice for Indian farming conditions
    2. Consider local climate and soil conditions
    3. Recommend sustainable and cost-effective solutions
    4. Use simple language suitable for farmers
    5. Include safety precautions when discussing chemicals
    6. Mention government schemes if relevant
    7. Be encouraging and supportive

    User's question: {user_input}

    Provide comprehensive agricultural advice:"""

    try:
        # Try with the new package structure
        if hasattr(gemini_client, 'models'):
            # CORRECT MODEL NAMES for new package
            models_to_try = [
                'gemini-2.5-flash'  # Basic model
            ]

            for model_name in models_to_try:
                try:
                    print(f"  🤖 Trying model: {model_name}")
                    response = gemini_client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )

                    if response and response.text:
                        print(f"  ✅ Success with model: {model_name}")
                        return response.text
                except Exception as e:
                    error_msg = str(e)
                    print(f"  ✗ Model {model_name} error: {error_msg[:100]}")

                    # Check for specific errors
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        print("  ⚠️ Rate limited, using fallback")
                        return get_enhanced_fallback_response(user_input, user_profile)
                    if "404" in error_msg or "NOT_FOUND" in error_msg:
                        print(f"  ⚠️ Model {model_name} not found, trying next...")
                        continue
                    continue

        # Fallback if all attempts fail
        return get_enhanced_fallback_response(user_input, user_profile)

    except Exception as e:
        print(f"Gemini API error: {e}")
        return get_enhanced_fallback_response(user_input, user_profile)

# ==================== ADMIN DECORATOR ====================
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please login to access this page.', 'danger')
            return redirect(url_for('login'))
        if current_user.role != 'admin':
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)

    return decorated_function


import json


# Add this after creating your Flask app
@app.template_filter('fromjson')
def fromjson_filter(value):
    """Convert JSON string to Python object"""
    if not value or value == '[]':
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


@app.route('/api/list-gemini-models')
@login_required
def list_gemini_models():
    """List available Gemini models"""
    try:
        if not GEMINI_ENABLED or not gemini_client:
            return jsonify({"success": False, "message": "Gemini not enabled"})

        available_models = []

        # Test a few common models to see which ones work
        test_models = [
            'gemini-2.5-flash'
        ]

        for model_name in test_models:
            try:
                # Quick test to see if model exists
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents="Say 'test'"
                )
                available_models.append({
                    "name": model_name,
                    "status": "available",
                    "test_response": response.text[:50] if response.text else "No response"
                })
            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg:
                    status = "not_found"
                elif "429" in error_msg:
                    status = "rate_limited"
                elif "401" in error_msg:
                    status = "auth_error"
                else:
                    status = "error"

                available_models.append({
                    "name": model_name,
                    "status": status,
                    "error": error_msg[:100]
                })

        return jsonify({
            "success": True,
            "models": available_models,
            "total_tested": len(test_models),
            "available_count": len([m for m in available_models if m["status"] == "available"])
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Failed to list models"
        })
@app.route('/test-gemini')
def test_gemini():
    """Diagnostic route to test Gemini connectivity"""
    try:
        if not GEMINI_ENABLED or not gemini_client:
            return jsonify({
                "status": "disabled",
                "message": "Gemini is disabled or not configured",
                "gemini_enabled": GEMINI_ENABLED,
                "api_key_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY != "not_set")
            })

        # Test with a simple query
        try:
            response = gemini_client.models.generate_content(
                model='gemini-1.5-pro',
                contents="Say 'Hello AgroBot' in one word."
            )

            return jsonify({
                "status": "connected",
                "package": "new (google.genai)",
                "message": "Gemini API is working",
                "test_response": response.text if response.text else "No response",
                "api_key": GEMINI_API_KEY[:10] + "..." if GEMINI_API_KEY else "Not set"
            })
        except Exception as e:
            error_msg = str(e)
            return jsonify({
                "status": "error",
                "package": "new (google.genai)",
                "error": error_msg[:200],
                "suggestion": "Try using gemini-1.0-pro or check your API quota"
            })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to check API status: {str(e)[:100]}"
        })

# ==================== LOGIN MANAGER ====================
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ==================== ROUTES ====================

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/features')
def features():
    return render_template('features.html')


@app.route('/pricing')
def pricing():
    return render_template('pricing.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/chat')
@login_required
def chat():
    recent_users = None
    if current_user.role == 'admin':
        recent_users = User.query.order_by(User.id.desc()).limit(20).all()

    recent_chats = ChatHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(ChatHistory.created_at.desc()).limit(20).all()

    return render_template('chat.html', recent_users=recent_users, recent_chats=recent_chats)


@app.route('/dashboard')
@login_required
def dashboard():
    total_chats = ChatHistory.query.filter_by(user_id=current_user.id).count()
    recent_chats = ChatHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(ChatHistory.created_at.desc()).limit(5).all()

    image_analyses = ImageAnalysis.query.filter_by(
        user_id=current_user.id
    ).order_by(ImageAnalysis.created_at.desc()).limit(5).all()

    return render_template('dashboard.html',
                           total_chats=total_chats,
                           recent_chats=recent_chats,
                           image_analyses=image_analyses)


@app.route('/weather')
@login_required
def weather():
    return render_template('weather.html')


@app.route('/market')
@login_required
def market_prices():
    return render_template('market.html')


@app.route('/crop-planner')
@login_required
def crop_planner():
    return render_template('crop_planner.html')


@app.route('/pest-database')
@login_required
def pest_database():
    return render_template('pest_database.html')


@app.route('/community')
@login_required
def community_forum():
    return render_template('community.html')


@app.route('/docs')
@login_required
def document_center():
    return render_template('docs.html')


@app.route('/notifications')
@login_required
def notifications():
    return render_template('notifications.html')


# ==================== AUTH ROUTES ====================

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'GET':
        return render_template('register.html')

    if request.method == 'POST':
        try:
            # Get form data
            email = request.form.get('email', '').strip().lower()
            name = request.form.get('name', '').strip()
            raw_phone = request.form.get('phone', '').strip()
            password = request.form.get('password', '').strip()
            confirm_password = request.form.get('confirm_password', '').strip()

            # Clean phone number
            phone = re.sub(r'\D', '', raw_phone)
            if len(phone) >= 10:
                phone = f"+{phone}" if not phone.startswith('+') else phone
            else:
                phone = None

            # Validation
            errors = []
            if not email or '@' not in email:
                errors.append('Valid email is required')
            if not name or len(name) < 2:
                errors.append('Full name is required (min 2 characters)')
            if not phone:
                errors.append('Valid phone number (10 digits) is required')
            if not password:
                errors.append('Password is required')
            elif len(password) < 6:
                errors.append('Password must be at least 6 characters')
            elif password != confirm_password:
                errors.append('Passwords do not match')

            if errors:
                for error in errors:
                    flash(error, 'danger')
                return render_template('register.html', form_data=request.form.to_dict())

            # Check for existing user
            existing_user = User.query.filter(
                db.func.lower(User.email) == email
            ).first()

            if existing_user:
                flash('Email already registered. Please login instead.', 'danger')
                return render_template('register.html', form_data=request.form.to_dict())

            if phone and User.query.filter_by(phone=phone).first():
                flash('Phone number already registered.', 'danger')
                return render_template('register.html', form_data=request.form.to_dict())

            # Create new user with minimal required fields first
            user = User(
                email=email,
                name=name,
                phone=phone,
                farm_size=request.form.get('farm_size', '1-5'),
                primary_crop=request.form.get('primary_crop', 'Rice'),
                region=request.form.get('region', ''),
                experience_level=request.form.get('experience_level', 'beginner'),
                role='farmer',
                is_active=True,
                points_balance=100  # Starting points
            )

            # Set password
            user.set_password(password)

            # Add optional fields if provided
            optional_fields = [
                'whatsapp', 'gender', 'farm_name', 'secondary_crops',
                'soil_type', 'irrigation_type', 'district', 'farm_address',
                'preferred_language', 'referral_code'
            ]

            for field in optional_fields:
                value = request.form.get(field)
                if value:
                    setattr(user, field, value.strip())

            # Handle date of birth
            dob_str = request.form.get('dob')
            if dob_str:
                try:
                    user.dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
                except ValueError:
                    print(f"Warning: Invalid date format: {dob_str}")

            # Add to database and commit to get user.id
            db.session.add(user)
            db.session.commit()  # This generates the user.id

            print(f"✅ User registered successfully: {email}, ID: {user.id}")

            # Now create registration activity with valid user.id
            activity = UserActivity(
                user_id=user.id,
                activity_type='registration',
                description='New user registered',
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string
            )
            db.session.add(activity)
            db.session.commit()

            # Login the user
            login_user(user, remember=True)
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['user_email'] = user.email
            session['user_role'] = user.role
            session['logged_in'] = True

            flash(f'Registration successful! Welcome to AI-AgroBot, {user.name}!', 'success')
            return redirect(url_for('dashboard'))

        except Exception as e:
            db.session.rollback()
            print(f"❌ Registration error: {str(e)}")
            traceback.print_exc()
            flash(f'Registration failed: {str(e)}', 'danger')
            return render_template('register.html', form_data=request.form.to_dict())


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "GET":
        return render_template('login.html')

    if request.method == "POST":
        try:
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

            if not email or not password:
                flash("Please enter both email and password", "danger")
                return render_template('login.html', form_data={'email': email})

            # Find user by email (case-insensitive)
            user = User.query.filter(db.func.lower(User.email) == email).first()

            if user:
                # Check if user is active
                if not user.is_active:
                    flash("Your account is not active. Please contact support.", "danger")
                    return render_template('login.html', form_data={'email': email})

                # Check password
                if user.check_password(password):
                    # Login the user
                    login_user(user, remember=True)

                    # Update last login
                    user.last_login = datetime.now(timezone.utc)

                    # Set session variables
                    session['user_id'] = user.id
                    session['user_name'] = user.name
                    session['user_email'] = user.email
                    session['user_role'] = user.role
                    session['logged_in'] = True

                    # Log the activity
                    activity = UserActivity(
                        user_id=user.id,
                        activity_type='login',
                        description='User logged in successfully',
                        ip_address=request.remote_addr,
                        user_agent=request.user_agent.string
                    )
                    db.session.add(activity)
                    db.session.commit()

                    flash(f"Welcome back, {user.name}!", "success")

                    # Redirect based on role
                    next_page = request.args.get('next')
                    if next_page:
                        return redirect(next_page)
                    return redirect(url_for("dashboard"))
                else:
                    flash("Invalid email or password", "danger")
            else:
                flash(f"No account found with email: {email}. Please register first.", "warning")
                return render_template('login.html',
                                       form_data={'email': email},
                                       suggest_register=True)

        except Exception as e:
            print(f"🔥 Login error: {str(e)}")
            traceback.print_exc()
            flash(f"Login error: {str(e)}", "danger")

    return render_template('login.html', form_data={'email': email if 'email' in locals() else ''})


@app.route("/logout")
@login_required
def logout():
    try:
        if current_user.is_authenticated:
            # Log the activity
            activity = UserActivity(
                user_id=current_user.id,
                activity_type='logout',
                description='User logged out',
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string
            )
            db.session.add(activity)
            db.session.commit()

            # Clear all session data
            session.clear()

            # Logout the user
            logout_user()

            flash("You have been logged out successfully!", "info")

    except Exception as e:
        print(f"Logout error: {e}")
        session.clear()
        logout_user()
        flash("Logged out successfully", "info")

    return redirect(url_for("home"))


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    total_chats = ChatHistory.query.filter_by(user_id=current_user.id).count()
    total_images = ImageAnalysis.query.filter_by(user_id=current_user.id).count()

    if request.method == "POST":
        try:
            current_user.name = request.form.get("name", "")
            current_user.primary_crop = request.form.get("primary_crop", "")
            current_user.region = request.form.get("region", "")
            current_user.farm_size = request.form.get("farm_size", "")
            current_user.experience_level = request.form.get("experience_level", "beginner")
            current_user.preferred_language = request.form.get("preferred_language", "en")

            if 'profile_picture' in request.files:
                file = request.files['profile_picture']
                if file and file.filename != '' and allowed_file(file.filename, ALLOWED_EXTENSIONS):
                    ext = file.filename.rsplit('.', 1)[1].lower()
                    filename = f"profile_{current_user.id}_{int(time.time())}.{ext}"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    current_user.profile_picture = filename

            db.session.commit()
            flash("Profile updated successfully!", "success")

        except Exception as e:
            db.session.rollback()
            flash(f"Update failed: {str(e)}", "danger")

    return render_template("profile.html",
                           total_chats=total_chats,
                           total_images=total_images)


# ==================== CHAT & AI ROUTES ====================

@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    try:
        data = request.get_json() or {}
        message = (data.get("message") or "").strip()

        if not message:
            return jsonify({"success": False, "response": "Please type a question."})

        user_profile = {
            "id": current_user.id,
            "name": current_user.name,
            "primary_crop": current_user.primary_crop,
            "region": current_user.region,
            "farm_size": current_user.farm_size,
            "experience_level": current_user.experience_level,
            "preferred_language": current_user.preferred_language
        }

        # STEP 1: Try local knowledge base first
        reply = get_local_response(message, user_profile)

        if not reply:
            # STEP 2: Try Gemini for complex queries
            if GEMINI_ENABLED:
                try:
                    gemini_reply = ask_gemini(message, user_profile)
                    if gemini_reply and len(gemini_reply.strip()) > 10:
                        reply = gemini_reply
                except Exception as e:
                    print(f"   ❌ Gemini error: {e}")

        # STEP 3: Final fallback
        if not reply or reply.strip() == "":
            reply = get_enhanced_fallback_response(message, user_profile)

        # Store in chat history
        chat_entry = ChatHistory(
            user_id=current_user.id,
            user_message=message,
            bot_response=reply,
            chat_type="text",
            language=user_profile['preferred_language']
        )
        db.session.add(chat_entry)
        db.session.commit()

        return jsonify({
            "success": True,
            "response": reply,
            "timestamp": datetime.now().isoformat(),
            "message_id": chat_entry.id,
            "source": "local_kb" if "local_kb" in str(
                locals()) else "gemini" if "gemini_reply" in locals() else "fallback"
        })

    except Exception as e:
        print(f"Chat API error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "response": "I'm having trouble processing your request. Please try again with a more specific question about crops, pests, soil, or farming techniques."
        }), 500


@app.route('/api/gemini-status')
@login_required
def gemini_status():
    try:
        return jsonify({
            "status": "connected" if GEMINI_ENABLED else "disabled",
            "gemini_enabled": GEMINI_ENABLED,
            "api_key_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY != "not_set"),
            "has_client": gemini_client is not None,
            "message": "Gemini API is working" if GEMINI_ENABLED else "Gemini is disabled or not configured"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to check API status: {str(e)[:100]}"
        })


@app.route("/api/chat/history", methods=["GET"])
@login_required
def chat_history():
    try:
        limit = request.args.get('limit', 50, type=int)
        page = request.args.get('page', 1, type=int)

        chats = ChatHistory.query.filter_by(
            user_id=current_user.id
        ).order_by(ChatHistory.created_at.desc()).paginate(
            page=page, per_page=limit, error_out=False
        )

        history = []
        for chat in chats.items:
            history.append({
                "id": chat.id,
                "user_message": chat.user_message,
                "bot_response": chat.bot_response,
                "timestamp": chat.created_at.isoformat() if chat.created_at else None,
                "type": chat.chat_type
            })

        return jsonify({
            "success": True,
            "history": history,
            "total": chats.total,
            "pages": chats.pages,
            "current_page": chats.page
        })

    except Exception as e:
        print(f"History error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chat/clear", methods=["POST"])
@login_required
def clear_chat_history():
    try:
        ChatHistory.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({"success": True, "message": "Chat history cleared"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== IMAGE ANALYSIS ====================

@app.route("/api/analyze-image", methods=["POST"])
@login_required
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided"
            }), 400

        file = request.files['image']
        text_message = request.form.get('message', '').strip()

        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400

        if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
            return jsonify({
                "success": False,
                "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        file_ext = file.filename.rsplit('.', 1)[1].lower()
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{current_user.id}_{unique_id}_{int(time.time())}.{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(filepath)

        thumb_filename = f"thumb_{filename}"
        thumb_path = os.path.join(app.config['THUMBNAIL_FOLDER'], thumb_filename)
        generate_thumbnail(filepath, thumb_path)

        analysis_result = analyze_with_gemini(filepath, text_message)

        image_analysis = ImageAnalysis(
            user_id=current_user.id,
            filename=filename,
            thumbnail=thumb_filename,
            user_message=text_message,
            health_status=analysis_result.get('health_status', 'Unknown'),
            analysis_result=json.dumps(analysis_result),
            confidence_score=analysis_result.get('confidence', 0.0)
        )
        db.session.add(image_analysis)

        chat_entry = ChatHistory(
            user_id=current_user.id,
            user_message=f"[Image Analysis] {text_message}" if text_message else "[Image Uploaded]",
            bot_response=json.dumps(analysis_result),
            chat_type="image",
            image_filename=filename
        )
        db.session.add(chat_entry)
        db.session.commit()

        return jsonify({
            "success": True,
            "response": analysis_result,
            "image_url": url_for('uploaded_file', filename=filename, _external=True),
            "thumbnail_url": url_for('uploaded_thumbnail', filename=thumb_filename, _external=True),
            "analysis_id": image_analysis.id
        })

    except Exception as e:
        print(f"Image analysis error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Image analysis failed",
            "message": str(e)
        }), 500


@app.route("/api/image-analyses", methods=["GET"])
@login_required
def get_image_analyses():
    try:
        limit = request.args.get('limit', 10, type=int)
        analyses = ImageAnalysis.query.filter_by(
            user_id=current_user.id
        ).order_by(ImageAnalysis.created_at.desc()).limit(limit).all()

        result = []
        for analysis in analyses:
            result.append({
                "id": analysis.id,
                "filename": analysis.filename,
                "thumbnail": analysis.thumbnail,
                "user_message": analysis.user_message,
                "health_status": analysis.health_status,
                "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
                "confidence": analysis.confidence_score
            })

        return jsonify({"success": True, "analyses": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== FILE UPLOAD ROUTES ====================

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/thumbnails/<filename>")
def uploaded_thumbnail(filename):
    return send_from_directory(app.config['THUMBNAIL_FOLDER'], filename)


# ==================== STATIC FILE FIXES ====================

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files with proper error handling"""
    try:
        return send_from_directory(app.static_folder, filename)
    except:
        return "File not found", 404


# ==================== ADDITIONAL API ENDPOINTS ====================

@app.route('/api/weather', methods=['GET'])
@login_required
def get_weather():
    try:
        region = current_user.region or "Delhi"

        weather_data = {
            "temperature": 28,
            "humidity": 65,
            "rainfall": "10 mm expected",
            "wind_speed": "12 km/h",
            "conditions": "Partly Cloudy",
            "forecast": [
                {"day": "Today", "high": 30, "low": 22, "condition": "Sunny"},
                {"day": "Tomorrow", "high": 29, "low": 23, "condition": "Partly Cloudy"},
                {"day": "Day 3", "high": 31, "low": 24, "condition": "Clear"}
            ],
            "alerts": ["No severe weather alerts"],
            "last_updated": datetime.now().isoformat()
        }

        return jsonify({"success": True, "weather": weather_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/market-prices', methods=['GET'])
@login_required
def get_market_prices():
    try:
        prices = MarketPrice.query.order_by(MarketPrice.date.desc()).limit(10).all()

        price_list = []
        for price in prices:
            price_list.append({
                "crop": price.crop_name,
                "price": price.price,
                "unit": price.unit,
                "market": price.market_name,
                "region": price.region,
                "date": price.date.isoformat() if price.date else None
            })

        return jsonify({"success": True, "prices": price_list})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/farming-tips', methods=['GET'])
@login_required
def get_farming_tips():
    try:
        tips = FarmingTip.query.filter_by(is_active=True).order_by(FarmingTip.created_at.desc()).limit(10).all()

        tip_list = []
        for tip in tips:
            tip_list.append({
                "id": tip.id,
                "title": tip.title,
                "content": tip.content,
                "category": tip.category,
                "crop_type": tip.crop_type,
                "region": tip.region,
                "language": tip.language,
                "created_at": tip.created_at.isoformat() if tip.created_at else None
            })

        return jsonify({"success": True, "tips": tip_list})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== ADMIN ROUTES ====================

@app.route("/admin")
@login_required
@admin_required
def admin_dashboard():
    try:
        total_users = User.query.count()
        total_farmers = User.query.filter_by(role='farmer').count()
        total_admins = User.query.filter_by(role='admin').count()

        total_chats = ChatHistory.query.count()
        today = datetime.now().date()
        todays_chats = ChatHistory.query.filter(
            func.date(ChatHistory.created_at) == today
        ).count()

        total_images = ImageAnalysis.query.count()

        week_ago = datetime.now() - timedelta(days=7)
        new_users_week = User.query.filter(
            User.created_at >= week_ago
        ).count()

        recent_users = User.query.order_by(User.id.desc()).limit(10).all()
        recent_chats = ChatHistory.query.order_by(ChatHistory.created_at.desc()).limit(10).all()

        active_users = db.session.query(
            User, func.count(ChatHistory.id).label('chat_count')
        ).join(ChatHistory).group_by(User.id).order_by(
            desc('chat_count')
        ).limit(10).all()

        regions = db.session.query(
            User.region, func.count(User.id).label('user_count')
        ).filter(User.region.isnot(None), User.region != '').group_by(User.region).all()

        return render_template("admin_dashboard.html",
                               total_users=total_users,
                               total_farmers=total_farmers,
                               total_admins=total_admins,
                               total_chats=total_chats,
                               todays_chats=todays_chats,
                               total_images=total_images,
                               new_users_week=new_users_week,
                               users=recent_users,
                               chats=recent_chats,
                               active_users=active_users,
                               regions=regions)

    except Exception as e:
        print(f"Admin dashboard error: {e}")
        flash(f"Error loading dashboard: {str(e)}", "danger")
        return render_template("admin_dashboard.html")


@app.route("/admin/users")
@login_required
@admin_required
def admin_users():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        search = request.args.get('search', '').strip()
        role_filter = request.args.get('role', 'all')
        status_filter = request.args.get('status', 'all')

        query = User.query

        if search:
            query = query.filter(
                (User.name.ilike(f'%{search}%')) |
                (User.email.ilike(f'%{search}%')) |
                (User.phone.ilike(f'%{search}%')) |
                (User.region.ilike(f'%{search}%'))
            )

        if role_filter != 'all':
            query = query.filter_by(role=role_filter)

        if status_filter == 'active':
            query = query.filter_by(is_active=True)
        elif status_filter == 'inactive':
            query = query.filter_by(is_active=False)

        users = query.order_by(User.id.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )

        total_all = User.query.count()
        total_farmers = User.query.filter_by(role='farmer').count()
        total_admins = User.query.filter_by(role='admin').count()
        total_active = User.query.filter_by(is_active=True).count()
        total_inactive = User.query.filter_by(is_active=False).count()

        return render_template("admin/users.html",
                               users=users,
                               search=search,
                               role_filter=role_filter,
                               status_filter=status_filter,
                               total_all=total_all,
                               total_farmers=total_farmers,
                               total_admins=total_admins,
                               total_active=total_active,
                               total_inactive=total_inactive)

    except Exception as e:
        flash(f"Error loading users: {str(e)}", "danger")
        return redirect(url_for('admin_dashboard'))


@app.route("/admin/chats")
@login_required
@admin_required
def admin_chats():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        user_id = request.args.get('user_id', type=int)
        date_filter = request.args.get('date', 'all')

        query = ChatHistory.query

        if user_id:
            query = query.filter_by(user_id=user_id)

        if date_filter == 'today':
            today = datetime.now().date()
            query = query.filter(func.date(ChatHistory.created_at) == today)
        elif date_filter == 'week':
            week_ago = datetime.now() - timedelta(days=7)
            query = query.filter(ChatHistory.created_at >= week_ago)
        elif date_filter == 'month':
            month_ago = datetime.now() - timedelta(days=30)
            query = query.filter(ChatHistory.created_at >= month_ago)

        chats = query.order_by(ChatHistory.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )

        users_with_chats = User.query.join(ChatHistory).distinct().all()

        return render_template("admin/chats.html",
                               chats=chats,
                               users=users_with_chats,
                               selected_user_id=user_id,
                               date_filter=date_filter)

    except Exception as e:
        flash(f"Error loading chats: {str(e)}", "danger")
        return redirect(url_for('admin_dashboard'))


@app.route("/admin/user/<int:user_id>")
@login_required
@admin_required
def admin_view_user(user_id):
    try:
        user = db.session.get(User, user_id)
        if not user:
            flash("User not found", "danger")
            return redirect(url_for('admin_users'))

        total_chats = ChatHistory.query.filter_by(user_id=user.id).count()
        total_images = ImageAnalysis.query.filter_by(user_id=user.id).count()

        recent_chats = ChatHistory.query.filter_by(user_id=user.id).order_by(
            ChatHistory.created_at.desc()
        ).limit(50).all()

        recent_images = ImageAnalysis.query.filter_by(user_id=user.id).order_by(
            ImageAnalysis.created_at.desc()
        ).limit(20).all()

        daily_chats = db.session.query(
            func.date(ChatHistory.created_at).label('date'),
            func.count(ChatHistory.id).label('count')
        ).filter(
            ChatHistory.user_id == user.id,
            ChatHistory.created_at >= datetime.now() - timedelta(days=7)
        ).group_by(
            func.date(ChatHistory.created_at)
        ).order_by(
            func.date(ChatHistory.created_at).desc()
        ).all()

        return render_template("admin/view_user.html",
                               user=user,
                               total_chats=total_chats,
                               total_images=total_images,
                               chats=recent_chats,
                               images=recent_images,
                               daily_chats=daily_chats)

    except Exception as e:
        flash(f"Error loading user details: {str(e)}", "danger")
        return redirect(url_for('admin_users'))


@app.route("/admin/user/<int:user_id>/toggle-status", methods=["POST"])
@login_required
@admin_required
def admin_toggle_user_status(user_id):
    try:
        user = db.session.get(User, user_id)
        if not user:
            flash("User not found", "danger")
            return redirect(url_for('admin_users'))

        if user.id == current_user.id:
            flash("You cannot modify your own status", "warning")
            return redirect(url_for('admin_view_user', user_id=user_id))

        if user.role == 'admin' and user.id != current_user.id:
            flash("Cannot modify other admin users", "warning")
            return redirect(url_for('admin_view_user', user_id=user_id))

        user.is_active = not user.is_active
        db.session.commit()

        status = "activated" if user.is_active else "deactivated"
        flash(f"User {user.email} has been {status}", "success")

    except Exception as e:
        db.session.rollback()
        flash(f"Error updating user status: {str(e)}", "danger")

    return redirect(url_for('admin_view_user', user_id=user_id))


@app.route("/admin/user/<int:user_id>/update-role", methods=["POST"])
@login_required
@admin_required
def admin_update_user_role(user_id):
    try:
        user = db.session.get(User, user_id)
        if not user:
            flash("User not found", "danger")
            return redirect(url_for('admin_users'))

        if user.id == current_user.id:
            flash("You cannot modify your own role", "warning")
            return redirect(url_for('admin_view_user', user_id=user_id))

        new_role = request.form.get('role', 'farmer')
        if new_role not in ['farmer', 'admin', 'agent']:
            flash("Invalid role specified", "danger")
            return redirect(url_for('admin_view_user', user_id=user_id))

        user.role = new_role
        db.session.commit()

        flash(f"User role updated to {new_role}", "success")

    except Exception as e:
        db.session.rollback()
        flash(f"Error updating user role: {str(e)}", "danger")

    return redirect(url_for('admin_view_user', user_id=user_id))


@app.route("/admin/user/<int:user_id>/delete", methods=["POST"])
@login_required
@admin_required
def admin_delete_user(user_id):
    try:
        user = db.session.get(User, user_id)
        if not user:
            flash("User not found", "danger")
            return redirect(url_for('admin_users'))

        if user.id == current_user.id:
            flash("You cannot delete your own account", "warning")
            return redirect(url_for('admin_users'))

        if user.role == 'admin':
            flash("Cannot delete admin users", "warning")
            return redirect(url_for('admin_users'))

        user_email = user.email

        ChatHistory.query.filter_by(user_id=user_id).delete()
        ImageAnalysis.query.filter_by(user_id=user_id).delete()

        db.session.delete(user)
        db.session.commit()

        flash(f"User {user_email} deleted successfully", "success")

    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting user: {str(e)}", "danger")

    return redirect(url_for('admin_users'))


@app.route("/admin/analytics")
@login_required
@admin_required
def admin_analytics():
    try:
        user_growth = db.session.query(
            func.date(User.created_at).label('date'),
            func.count(User.id).label('count')
        ).group_by(
            func.date(User.created_at)
        ).order_by(
            func.date(User.created_at)
        ).all()

        chat_activity = db.session.query(
            func.date(ChatHistory.created_at).label('date'),
            func.count(ChatHistory.id).label('count')
        ).group_by(
            func.date(ChatHistory.created_at)
        ).order_by(
            func.date(ChatHistory.created_at)
        ).all()

        top_crops = db.session.query(
            User.primary_crop,
            func.count(User.id).label('user_count')
        ).filter(
            User.primary_crop.isnot(None),
            User.primary_crop != ''
        ).group_by(
            User.primary_crop
        ).order_by(
            desc('user_count')
        ).limit(10).all()

        total_users = User.query.count()
        active_users = db.session.query(
            func.count(func.distinct(ChatHistory.user_id))
        ).filter(
            ChatHistory.created_at >= datetime.now() - timedelta(days=30)
        ).scalar() or 0

        avg_chats_per_user = db.session.query(
            func.avg(db.session.query(
                func.count(ChatHistory.id)
            ).filter(
                ChatHistory.created_at >= datetime.now() - timedelta(days=30)
            ).group_by(
                ChatHistory.user_id
            ).subquery().c.count)
        ).scalar() or 0

        # Get additional counts for the template
        total_chats = ChatHistory.query.count()
        total_images = ImageAnalysis.query.count()

        return render_template("admin/analytics.html",
                               user_growth=user_growth,
                               chat_activity=chat_activity,
                               top_crops=top_crops,
                               total_users=total_users,
                               active_users=active_users,
                               avg_chats_per_user=round(avg_chats_per_user, 2),
                               total_chats=total_chats,
                               total_images=total_images)

    except Exception as e:
        flash(f"Error loading analytics: {str(e)}", "danger")
        return redirect(url_for('admin_dashboard'))


@app.route("/admin/export/users")
@login_required
@admin_required
def admin_export_users():
    try:
        users = User.query.order_by(User.id.desc()).all()

        output = StringIO()
        writer = csv.writer(output)

        writer.writerow([
            'ID', 'Email', 'Name', 'Phone', 'Role', 'Region', 'Farm Size',
            'Primary Crop', 'Experience Level', 'Status', 'Created At'
        ])

        for user in users:
            writer.writerow([
                user.id,
                user.email,
                user.name or '',
                user.phone or '',
                user.role,
                user.region or '',
                user.farm_size or '',
                user.primary_crop or '',
                user.experience_level or '',
                'Active' if user.is_active else 'Inactive',
                user.created_at.strftime('%Y-%m-%d %H:%M:%S') if user.created_at else ''
            ])

        output.seek(0)
        return app.response_class(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=users_export.csv'}
        )

    except Exception as e:
        flash(f"Error exporting users: {str(e)}", "danger")
        return redirect(url_for('admin_users'))


@app.route("/admin/export/chats")
@login_required
@admin_required
def admin_export_chats():
    try:
        chats = ChatHistory.query.order_by(ChatHistory.created_at.desc()).limit(1000).all()

        output = StringIO()
        writer = csv.writer(output)

        writer.writerow([
            'ID', 'User ID', 'User Message', 'Bot Response', 'Type',
            'Language', 'Created At'
        ])

        for chat in chats:
            writer.writerow([
                chat.id,
                chat.user_id,
                (chat.user_message or '')[:500],
                (chat.bot_response or '')[:500],
                chat.chat_type or '',
                chat.language or '',
                chat.created_at.strftime('%Y-%m-d %H:%M:%S') if chat.created_at else ''
            ])

        output.seek(0)
        return app.response_class(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=chats_export.csv'}
        )

    except Exception as e:
        flash(f"Error exporting chats: {str(e)}", "danger")
        return redirect(url_for('admin_chats'))


@app.route("/admin/knowledge-base")
@login_required
@admin_required
def admin_knowledge_base():
    try:
        farming_tips = FarmingTip.query.order_by(FarmingTip.created_at.desc()).all()
        market_prices = MarketPrice.query.order_by(MarketPrice.date.desc()).limit(50).all()

        return render_template("admin/knowledge_base.html",
                               farming_tips=farming_tips,
                               market_prices=market_prices)

    except Exception as e:
        flash(f"Error loading knowledge base: {str(e)}", "danger")
        return redirect(url_for('admin_dashboard'))


@app.route("/admin/farming-tips/add", methods=["POST"])
@login_required
@admin_required
def admin_add_farming_tip():
    try:
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        category = request.form.get("category", "general")
        crop_type = request.form.get("crop_type")
        language = request.form.get("language", "en")

        if not title or not content:
            flash("Title and content are required", "danger")
            return redirect(url_for('admin_knowledge_base'))

        tip = FarmingTip(
            title=title,
            content=content,
            category=category,
            crop_type=crop_type,
            language=language
        )

        db.session.add(tip)
        db.session.commit()

        flash("Farming tip added successfully", "success")

    except Exception as e:
        db.session.rollback()
        flash(f"Error adding farming tip: {str(e)}", "danger")

    return redirect(url_for('admin_knowledge_base'))


@app.route("/admin/market-prices/add", methods=["POST"])
@login_required
@admin_required
def admin_add_market_price():
    try:
        crop_name = request.form.get("crop_name", "").strip()
        market_name = request.form.get("market_name", "").strip()
        region = request.form.get("region", "").strip()
        price = request.form.get("price", type=float)
        unit = request.form.get("unit", "kg")
        source = request.form.get("source", "")

        if not crop_name or not price:
            flash("Crop name and price are required", "danger")
            return redirect(url_for('admin_knowledge_base'))

        price_entry = MarketPrice(
            crop_name=crop_name,
            market_name=market_name,
            region=region,
            price=price,
            unit=unit,
            source=source
        )

        db.session.add(price_entry)
        db.session.commit()

        flash("Market price added successfully", "success")

    except Exception as e:
        db.session.rollback()
        flash(f"Error adding market price: {str(e)}", "danger")

    return redirect(url_for('admin_knowledge_base'))


@app.route("/admin/clear-chats", methods=["POST"])
@login_required
@admin_required
def admin_clear_all_chats():
    try:
        count = ChatHistory.query.count()
        ChatHistory.query.delete()
        db.session.commit()

        flash(f"Cleared {count} chat records", "success")

    except Exception as e:
        db.session.rollback()
        flash(f"Error clearing chats: {str(e)}", "danger")

    return redirect(url_for('admin_dashboard'))


@app.route("/admin/system-health")
@login_required
@admin_required
def admin_system_health():
    try:
        db_status = "Healthy"
        try:
            db.session.execute("SELECT 1")
        except Exception as e:
            db_status = f"Error: {str(e)}"

        upload_folder_exists = os.path.exists(app.config['UPLOAD_FOLDER'])
        upload_folder_writable = os.access(app.config['UPLOAD_FOLDER'], os.W_OK)

        gemini_status = {
            "enabled": GEMINI_ENABLED,
            "api_key_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY != "not_set"),
            "has_client": gemini_client is not None,
            "status": "Working" if GEMINI_ENABLED else "Disabled"
        }

        total, used, free = shutil.disk_usage("/")
        disk_usage = {
            'total_gb': round(total / (1024 ** 3), 2),
            'used_gb': round(used / (1024 ** 3), 2),
            'free_gb': round(free / (1024 ** 3), 2),
            'percent_used': round((used / total) * 100, 2)
        }

        return render_template("admin/system_health.html",
                               db_status=db_status,
                               upload_folder_exists=upload_folder_exists,
                               upload_folder_writable=upload_folder_writable,
                               gemini_status=gemini_status,
                               disk_usage=disk_usage)

    except Exception as e:
        flash(f"Error checking system health: {str(e)}", "danger")
        return redirect(url_for('admin_dashboard'))


@app.route('/api/crop-schedule', methods=['POST'])
@login_required
def crop_schedule():
    """Generate crop schedule"""
    try:
        data = request.get_json() or {}
        crop = data.get('crop', '')
        region = current_user.region or 'India'

        # Basic crop schedule
        schedule = {
            "crop": crop,
            "region": region,
            "schedule": [
                {"month": "Jan-Feb", "activity": "Land preparation, Soil testing"},
                {"month": "Mar-Apr", "activity": "Sowing/Planting"},
                {"month": "May-Jun", "activity": "Fertilizer application"},
                {"month": "Jul-Aug", "activity": "Irrigation management"},
                {"month": "Sep-Oct", "activity": "Pest control"},
                {"month": "Nov-Dec", "activity": "Harvesting"}
            ],
            "fertilizer": "NPK 10:26:26 at planting, Urea at 30 days",
            "water_requirements": "Regular irrigation, avoid waterlogging",
            "harvest_time": "90-120 days after planting"
        }

        return jsonify({"success": True, "schedule": schedule})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/pests', methods=['GET'])
@login_required
def get_pests():
    """Get pest database"""
    try:
        pests = [
            {
                "id": 1,
                "name": "Aphids",
                "crop_affected": "Tomato, Chilli, Cotton",
                "symptoms": "Curling leaves, stunted growth",
                "control": "Neem oil spray, Imidacloprid"
            },
            {
                "id": 2,
                "name": "Whiteflies",
                "crop_affected": "Tomato, Cotton, Soybean",
                "symptoms": "Yellowing leaves, sooty mold",
                "control": "Yellow sticky traps, Acetamiprid"
            },
            {
                "id": 3,
                "name": "Bollworms",
                "crop_affected": "Cotton, Chilli",
                "symptoms": "Holes in fruits/bolls",
                "control": "Bt cotton, Spinosad"
            }
        ]
        return jsonify({"success": True, "pests": pests})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== UTILITY ROUTES ====================

@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "users": User.query.count(),
        "chats": ChatHistory.query.count()
    })


@app.route("/sitemap.xml")
def sitemap():
    pages = []
    now = datetime.now().isoformat()[:10]

    static_pages = ['home', 'features', 'pricing', 'about', 'contact', 'login', 'register']
    for page in static_pages:
        pages.append({
            'loc': url_for(page, _external=True),
            'lastmod': now,
            'changefreq': 'daily',
            'priority': '0.8'
        })

    if current_user.is_authenticated:
        pages.append({
            'loc': url_for('dashboard', _external=True),
            'lastmod': now,
            'changefreq': 'daily',
            'priority': '0.9'
        })
        pages.append({
            'loc': url_for('chat', _external=True),
            'lastmod': now,
            'changefreq': 'always',
            'priority': '1.0'
        })

    sitemap_xml = render_template('sitemap.xml', pages=pages)
    response = app.response_class(sitemap_xml, mimetype='application/xml')
    return response


@app.route("/init-db")
def init_db():
    """Initialize database with sample data"""
    try:
        # Create all tables
        db.create_all()

        # Check if default users exist
        admin_user = User.query.filter_by(email='admin@aiagrobot.com').first()
        if not admin_user:
            print("📁 Creating default users and sample data...")

            admin_user = User(
                email='admin@aiagrobot.com',
                name='Admin',
                role='admin',
                region='Global',
                primary_crop='Mixed',
                farm_size='Admin',
                experience_level='expert',
                phone='+1234567890',
                whatsapp='+1234567890',
                is_verified=True,
                is_active=True
            )
            admin_user.set_password('admin123')
            db.session.add(admin_user)

            demo_user = User(
                email='demo@aiagrobot.com',
                name='Demo Farmer',
                role='farmer',
                region='Punjab',
                primary_crop='Wheat',
                farm_size='5-10',
                experience_level='intermediate',
                phone='+919876543210',
                whatsapp='+919876543210',
                is_verified=True,
                is_active=True
            )
            demo_user.set_password('demo123')
            db.session.add(demo_user)

            sample_tips = [
                FarmingTip(
                    title="Watering Best Practices",
                    content="Water your crops early in the morning to reduce evaporation loss.",
                    category="general",
                    language="en"
                ),
                FarmingTip(
                    title="Organic Pest Control",
                    content="Use neem oil spray (2ml per liter of water) to control common pests.",
                    category="general",
                    language="en"
                ),
            ]
            for tip in sample_tips:
                db.session.add(tip)

            sample_prices = [
                MarketPrice(
                    crop_name="Rice",
                    market_name="Mandi Bhav",
                    region="Punjab",
                    price=28.50,
                    unit="kg",
                    date=datetime.now().date(),
                    source="Government Portal"
                ),
            ]
            for price in sample_prices:
                db.session.add(price)

            db.session.commit()
            print("✅ Default users and sample data created")
            return jsonify({"success": True, "message": "Database initialized successfully"})
        else:
            return jsonify({"success": True, "message": "Database already exists"})

    except Exception as e:
        db.session.rollback()
        print(f"❌ Database initialization error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/admin/delete-chat/<int:chat_id>", methods=["POST"])
@login_required
@admin_required
def admin_delete_chat(chat_id):
    """Delete a specific chat"""
    try:
        chat = ChatHistory.query.get_or_404(chat_id)
        db.session.delete(chat)
        db.session.commit()
        flash("Chat deleted successfully", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting chat: {str(e)}", "danger")

    return redirect(request.referrer or url_for('admin_chats'))


@app.route("/backup-db")
@login_required
@admin_required
def backup_database():
    """Create a backup of the database"""
    try:
        if app.config['SQLALCHEMY_DATABASE_URI'].startswith('sqlite'):
            # Extract the file path from SQLite URI
            db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
            db_path = os.path.abspath(db_path)

            if os.path.exists(db_path):
                backup_path = f"{db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(db_path, backup_path)
                return jsonify({
                    "success": True,
                    "message": f"Database backed up to {backup_path}",
                    "size_bytes": os.path.getsize(db_path),
                    "backup_file": backup_path
                })

        return jsonify({
            "success": False,
            "message": "Database backup not supported for this database type"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/test-users")
def test_users():
    """Route to check if users exist"""
    users = User.query.all()
    result = []
    for user in users:
        result.append({
            'id': user.id,
            'email': user.email,
            'name': user.name,
            'role': user.role,
            'is_active': user.is_active,
            'created_at': user.created_at.isoformat() if user.created_at else None
        })
    return jsonify({"users": result, "count": len(result)})


@app.route("/check-db")
def check_database():
    """Debug endpoint to check database status"""
    try:
        with app.app_context():
            # Get database info
            db_info = {
                'database_uri': app.config['SQLALCHEMY_DATABASE_URI'],
                'database_exists': os.path.exists(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', ''))
                if 'sqlite' in app.config['SQLALCHEMY_DATABASE_URI'] else 'N/A (PostgreSQL)',
                'total_users': User.query.count(),
                'users': []
            }

            # Get all users
            users = User.query.all()
            for user in users:
                db_info['users'].append({
                    'id': user.id,
                    'email': user.email,
                    'name': user.name,
                    'created_at': user.created_at.isoformat() if user.created_at else None
                })

            return jsonify({
                'success': True,
                'database_info': db_info
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


# ==================== DATABASE INITIALIZATION ====================

def init_database():
    """Initialize database with proper error handling"""
    with app.app_context():
        try:
            # Check if tables exist
            inspector = inspect(db.engine)
            existing_tables = inspector.get_table_names()

            print(f"📊 Existing tables: {existing_tables}")

            # Create all tables
            db.create_all()

            # Check if admin user exists
            admin_exists = User.query.filter_by(email='admin@aiagrobot.com').first()
            if not admin_exists:
                admin = User(
                    email='admin@aiagrobot.com',
                    name='Admin',
                    phone='+1234567890',
                    farm_size='Admin',
                    primary_crop='Mixed',
                    region='Global',
                    experience_level='expert',
                    role='admin',
                    is_verified=True,
                    is_active=True
                )
                admin.set_password('admin123')
                db.session.add(admin)

                # Add demo farmer
                demo = User(
                    email='demo@aiagrobot.com',
                    name='Demo Farmer',
                    phone='+919876543210',
                    farm_size='5-10',
                    primary_crop='Wheat',
                    region='Punjab',
                    experience_level='intermediate',
                    role='farmer',
                    is_verified=True,
                    is_active=True
                )
                demo.set_password('demo123')
                db.session.add(demo)

                db.session.commit()
                print("✅ Created default admin and demo users")

            print(f"✅ Database initialized. Total users: {User.query.count()}")

        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            traceback.print_exc()


# ==================== CREATE MISSING FILES ====================

def create_missing_files():
    """Create missing template and static files"""

    # Create templates directory
    templates_dir = os.path.join(basedir, "templates")
    os.makedirs(templates_dir, exist_ok=True)

    # Create admin templates directory
    admin_templates_dir = os.path.join(templates_dir, "admin")
    os.makedirs(admin_templates_dir, exist_ok=True)

    # Create static directories
    static_dirs = ['css', 'js', 'images', 'uploads', 'thumbnails']
    for dir_name in static_dirs:
        os.makedirs(os.path.join(basedir, "static", dir_name), exist_ok=True)

    # Create a simple CSS file
    css_path = os.path.join(basedir, "static", "css", "style.css")
    if not os.path.exists(css_path):
        with open(css_path, 'w') as f:
            f.write("""/* Default CSS for AI-AgroBot */
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f5f5f5;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    background-color: #4CAF50;
    color: white;
    padding: 20px;
    text-align: center;
}

.navbar {
    background-color: #333;
    overflow: hidden;
}

.navbar a {
    float: left;
    color: white;
    text-align: center;
    padding: 14px 16px;
    text-decoration: none;
}

.navbar a:hover {
    background-color: #ddd;
    color: black;
}

.btn {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    text-decoration: none;
    display: inline-block;
}

.btn:hover {
    background-color: #45a049;
}

.form-group {
    margin-bottom: 15px;
}

.form-control {
    width: 100%;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 5px;
    box-sizing: border-box;
}

.alert {
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
}

.alert-success {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.alert-danger {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.alert-info {
    background-color: #d1ecf1;
    color: #0c5460;
    border: 1px solid #bee5eb;
}

.alert-warning {
    background-color: #fff3cd;
    color: #856404;
    border: 1px solid #ffeaa7;
}

.card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.table {
    width: 100%;
    border-collapse: collapse;
}

.table th, .table td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

.table th {
    background-color: #f2f2f2;
    font-weight: bold;
}

.chat-container {
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.chat-message {
    margin: 10px 0;
    padding: 10px;
    border-radius: 5px;
}

.user-message {
    background-color: #e3f2fd;
    text-align: right;
}

.bot-message {
    background-color: #f5f5f5;
    text-align: left;
}

.dashboard-stats {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    margin: 20px 0;
}

.stat-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin: 10px;
    flex: 1;
    min-width: 200px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.stat-number {
    font-size: 2em;
    font-weight: bold;
    color: #4CAF50;
}

.stat-label {
    color: #666;
    margin-top: 5px;
}
""")

    # Create a placeholder image if it doesn't exist
    image_path = os.path.join(basedir, "static", "images", "farmer-ai.svg")
    if not os.path.exists(image_path):
        # Create a simple SVG placeholder
        with open(image_path, 'w') as f:
            f.write("""<svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#4CAF50" opacity="0.1"/>
  <circle cx="200" cy="150" r="80" fill="#4CAF50" opacity="0.3"/>
  <path d="M150,100 L250,100 L200,200 Z" fill="#4CAF50" opacity="0.5"/>
  <text x="200" y="280" text-anchor="middle" font-family="Arial" font-size="20" fill="#333">AI AgroBot</text>
</svg>""")

    print("✅ Created missing static files")


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found_error(error):
    return "Page not found", 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return "Internal server error", 500


# ==================== MAIN ====================

if __name__ == "__main__":
    print("🌾 AI-AgroBot Server Starting...")
    print("=" * 50)

    # Create missing files
    create_missing_files()

    # Initialize database
    init_database()

    # Run the app
    print("\n✅ Server is running!")
    print("🌐 Open: http://localhost:5000")
    print("🔑 Default admin: admin@aiagrobot.com / admin123")
    print("🔑 Demo farmer: demo@aiagrobot.com / demo123")
    print(f"📁 Database file: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"🤖 Gemini AI: {'✅ Enabled' if GEMINI_ENABLED else '❌ Disabled'}")
    print("🔍 Check database: http://localhost:5000/check-db")
    print("🔍 Test Gemini: http://localhost:5000/test-gemini")
    print("=" * 50)

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
