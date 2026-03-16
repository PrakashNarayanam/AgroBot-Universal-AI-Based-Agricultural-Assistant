import os
import json
import time
import traceback
import csv
import uuid
import re
import shutil
import base64
import logging
from datetime import datetime, timedelta, timezone
from io import StringIO
from functools import wraps

import requests
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    jsonify, send_from_directory, session, send_file, abort, current_app
)
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import func, desc, text, inspect
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from flask_socketio import SocketIO, emit, join_room, leave_room
from datetime import datetime, date, timedelta, timezone
load_dotenv()

# -------------------------------------------------------------------
# 1. Create Flask app FIRST
# -------------------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")

# -------------------------------------------------------------------
# 2. Configuration
# -------------------------------------------------------------------
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "super_secret_key_123")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ECHO"] = False
app.config['WEATHER_API_KEY'] = '72b6594b33a820f2fa2df11ca725b2c3'
app.config['WEATHER_API_URL'] = 'https://api.openweathermap.org/data/2.5'

# Database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
database_url = os.getenv('DATABASE_URL')

if database_url:
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    print(f"📁 Using PostgreSQL database")
else:
    db_dir = os.path.join(basedir, "data")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "agrobot.db")
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    print(f"📁 SQLite Database file: {db_path}")

# File upload configuration
UPLOAD_FOLDER = os.path.join(basedir, "static", "uploads")
THUMBNAIL_FOLDER = os.path.join(basedir, "static", "thumbnails")
DOCUMENT_FOLDER = os.path.join(basedir, "uploads", "documents")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["THUMBNAIL_FOLDER"] = THUMBNAIL_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER_DOCS'] = DOCUMENT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)
os.makedirs(DOCUMENT_FOLDER, exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_DOC_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx', 'csv'}

# -------------------------------------------------------------------
# 3. Initialize extensions (db, login_manager)
# -------------------------------------------------------------------
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# -------------------------------------------------------------------
# 4. Initialize SocketIO (app is defined)
# -------------------------------------------------------------------
socketio = SocketIO(app, cors_allowed_origins="*")

# -------------------------------------------------------------------
# 5. Gemini AI setup
# -------------------------------------------------------------------
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_ENABLED = False
gemini_client = None

try:
    from google import genai as google_genai
    if GEMINI_API_KEY and GEMINI_API_KEY != "not_set":
        gemini_client = google_genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_ENABLED = True
        print("🔑 Gemini configured (new package)")
except ImportError:
    try:
        import google.generativeai as genai_old
        if GEMINI_API_KEY and GEMINI_API_KEY != "not_set":
            genai_old.configure(api_key=GEMINI_API_KEY)
            gemini_client = genai_old
            GEMINI_ENABLED = True
            print("🔑 Gemini configured (old package)")
    except ImportError:
        print("⚠️ No Gemini package found")
except Exception as e:
    print(f"⚠️ Gemini configuration failed: {e}")

if not GEMINI_API_KEY or GEMINI_API_KEY == "not_set":
    print("⚠️ Warning: GEMINI_API_KEY not found in .env file")

# -------------------------------------------------------------------
# Helper function for UTC timestamps
# -------------------------------------------------------------------
def utc_now():
    return datetime.now(timezone.utc)

# -------------------------------------------------------------------
# DATABASE MODELS
# -------------------------------------------------------------------

class User(db.Model, UserMixin):
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
    notify_weather = db.Column(db.Boolean, default=True)
    notify_pests = db.Column(db.Boolean, default=True)
    notify_market = db.Column(db.Boolean, default=True)
    notify_tips = db.Column(db.Boolean, default=True)
    interest_organic = db.Column(db.Boolean, default=False)
    interest_hydroponics = db.Column(db.Boolean, default=False)
    interest_precision = db.Column(db.Boolean, default=False)
    interest_dairy = db.Column(db.Boolean, default=False)
    interest_poultry = db.Column(db.Boolean, default=False)
    interest_fisheries = db.Column(db.Boolean, default=False)
    newsletter = db.Column(db.Boolean, default=True)
    share_data = db.Column(db.Boolean, default=False)
    referral_code = db.Column(db.String(50))
    points_balance = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    role = db.Column(db.String(20), default='farmer')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))
    last_login = db.Column(db.DateTime)
    email_verified_at = db.Column(db.DateTime)
    phone_verified_at = db.Column(db.DateTime)

    chats = db.relationship('ChatHistory', backref='user', lazy=True, cascade='all, delete-orphan')
    image_analyses = db.relationship('ImageAnalysis', backref='user', lazy=True, cascade='all, delete-orphan')
    activities = db.relationship('UserActivity', backref='user', lazy=True, cascade='all, delete-orphan')
    points = db.relationship('UserPoints', backref='user', lazy=True, cascade='all, delete-orphan')
    crop_plans = db.relationship('CropPlan', backref='user', lazy=True, cascade='all, delete-orphan')
    forum_threads = db.relationship('ForumThread', backref='user', lazy=True)
    forum_posts = db.relationship('ForumPost', backref='user', lazy=True)
    documents = db.relationship('Document', backref='user', lazy='dynamic')

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

    def get_id(self):
        return str(self.id)

    def to_dict(self):
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
    __tablename__ = 'user_activities'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    activity_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class UserPoints(db.Model):
    __tablename__ = 'user_points'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    points = db.Column(db.Integer, nullable=False)
    balance_after = db.Column(db.Integer, nullable=False)
    transaction_type = db.Column(db.String(20), nullable=False)
    reason = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class Referral(db.Model):
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


class CropPlan(db.Model):
    __tablename__ = 'crop_plans'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    crop_type = db.Column(db.String(100), nullable=False)
    variety = db.Column(db.String(100))
    start_date = db.Column(db.Date, nullable=False)
    expected_harvest = db.Column(db.Date, nullable=False)
    area = db.Column(db.Float)
    planting_method = db.Column(db.String(50))
    notes = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    tasks = db.relationship('CropTask', backref='plan', lazy=True, cascade='all, delete-orphan')


class CropTask(db.Model):
    __tablename__ = 'crop_tasks'

    id = db.Column(db.Integer, primary_key=True)
    plan_id = db.Column(db.Integer, db.ForeignKey('crop_plans.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    due_date = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(20), default='pending')
    category = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime)


# ==================== FORUM MODELS ====================

class ForumCategory(db.Model):
    __tablename__ = 'forum_categories'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(255))
    icon = db.Column(db.String(50))
    color = db.Column(db.String(20))
    thread_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    threads = db.relationship('ForumThread', backref='category', lazy=True)


class ForumThread(db.Model):
    __tablename__ = 'forum_threads'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('forum_categories.id'), nullable=False)
    views = db.Column(db.Integer, default=0)
    is_pinned = db.Column(db.Boolean, default=False)
    is_locked = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    posts = db.relationship('ForumPost', backref='thread', lazy=True, cascade='all, delete-orphan')


class ForumPost(db.Model):
    __tablename__ = 'forum_posts'
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    thread_id = db.Column(db.Integer, db.ForeignKey('forum_threads.id'), nullable=False)
    is_solution = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    likes = db.relationship('ForumLike', backref='post', lazy=True, cascade='all, delete-orphan')


class ForumLike(db.Model):
    __tablename__ = 'forum_likes'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('forum_posts.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    __table_args__ = (db.UniqueConstraint('user_id', 'post_id', name='unique_user_post_like'),)


class ForumTag(db.Model):
    __tablename__ = 'forum_tags'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)


class ForumThreadTag(db.Model):
    __tablename__ = 'forum_thread_tags'
    thread_id = db.Column(db.Integer, db.ForeignKey('forum_threads.id'), primary_key=True)
    tag_id = db.Column(db.Integer, db.ForeignKey('forum_tags.id'), primary_key=True)


class UserFollow(db.Model):
    """User follow relationships (many-to-many self-referential)"""
    __tablename__ = 'user_follows'

    id = db.Column(db.Integer, primary_key=True)
    follower_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    followed_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (db.UniqueConstraint('follower_id', 'followed_id', name='unique_follow'),)

    follower = db.relationship('User', foreign_keys=[follower_id], backref='following')
    followed = db.relationship('User', foreign_keys=[followed_id], backref='followers')


class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'

    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    room = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    file_url = db.Column(db.String(500))
    file_type = db.Column(db.String(50))
    reply_to_id = db.Column(db.Integer, db.ForeignKey('chat_messages.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    sender = db.relationship('User', foreign_keys=[sender_id])
    replies = db.relationship('ChatMessage', backref=db.backref('reply_to', remote_side=[id]))


class MessageReaction(db.Model):
    """Reactions (emojis) on chat messages"""
    __tablename__ = 'message_reactions'

    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(db.Integer, db.ForeignKey('chat_messages.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    emoji = db.Column(db.String(10), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (db.UniqueConstraint('message_id', 'user_id', 'emoji', name='unique_reaction'),)

    # This backref creates a `reactions` attribute on ChatMessage
    message = db.relationship('ChatMessage', backref='reactions')
    user = db.relationship('User', backref='reactions')


class PrivateMessage(db.Model):
    """Private messages between users"""
    __tablename__ = 'private_messages'

    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    recipient_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_messages')
    recipient = db.relationship('User', foreign_keys=[recipient_id], backref='received_messages')


# ==================== DOCUMENT MODEL ====================

class Document(db.Model):
    __tablename__ = 'documents'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50))
    size = db.Column(db.Integer)
    description = db.Column(db.Text, default='')
    category = db.Column(db.String(50), default='uploads')
    uploaded_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    def formatted_size(self):
        size = self.size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    @property
    def path(self):
        return os.path.join(current_app.config['UPLOAD_FOLDER_DOCS'], self.filename)

    @property
    def url(self):
        return f'/uploads/documents/{self.filename}'


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def clean_phone_number(phone):
    if not phone:
        return None
    digits = re.sub(r'\D', '', phone)
    if len(digits) >= 10:
        return f"+{digits}" if not digits.startswith('+') else digits
    return None


def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


def generate_thumbnail(image_path, thumb_path, size=(200, 200)):
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img.thumbnail(size)
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
    try:
        user_input_lower = user_input.lower()
        for topic, details in LOCAL_KNOWLEDGE.items():
            if topic in user_input_lower:
                for subtopic, response in details.items():
                    if subtopic in user_input_lower:
                        return f"**{topic.capitalize()} {subtopic.capitalize()}:**\n{response}\n\n*Source: Local Agricultural Knowledge Base*"
                first_key = next(iter(details))
                return f"**{topic.capitalize()} Information:**\n{details[first_key]}\n\nFor more specific advice, please ask about planting, fertilizer, water, or harvest."
        if any(word in user_input_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm AgroBot, your agricultural assistant. How can I help you with farming today?"
        if any(word in user_input_lower for word in ["help", "what can you do"]):
            return "I can help with:\n• Crop cultivation advice\n• Pest and disease identification\n• Soil and fertilizer recommendations\n• Irrigation guidance\n• Market price information\n• Weather impacts\n• Harvesting techniques\n\nJust ask me anything about farming!"
        return None
    except Exception as e:
        print(f"Error in get_local_response: {e}")
        return None


def get_enhanced_fallback_response(user_input, user_profile):
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
    if not GEMINI_ENABLED or not gemini_client:
        return fallback_image_analysis(image_path)

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        ext = image_path.lower()
        if ext.endswith('.png'):
            mime_type = 'image/png'
        elif ext.endswith('.jpg') or ext.endswith('.jpeg'):
            mime_type = 'image/jpeg'
        elif ext.endswith('.gif'):
            mime_type = 'image/gif'
        elif ext.endswith('.bmp'):
            mime_type = 'image/bmp'
        elif ext.endswith('.webp'):
            mime_type = 'image/webp'
        else:
            mime_type = 'image/jpeg'

        prompt = f"Analyze this agricultural image. User query: {user_message}\n\nProvide:\n1. Plant/crop identification\n2. Health assessment\n3. Pest/disease detection if any\n4. Recommendations" if user_message else \
                 "Analyze this agricultural image. Provide:\n1. Plant/crop identification\n2. Health assessment\n3. Pest/disease detection if any\n4. Recommendations"

        parts = [{"text": prompt}, {"inline_data": {"mime_type": mime_type, "data": image_b64}}]

        models_to_try = ['gemini-2.5-flash,gemini-1.5-flash']
        for model_name in models_to_try:
            try:
                print(f"  🤖 Trying vision model: {model_name}")
                response = gemini_client.models.generate_content(model=model_name, contents=parts)
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
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print("  ⚠️ Rate limited, using fallback analysis")
                    return fallback_image_analysis(image_path)
                if "404" in error_msg or "NOT_FOUND" in error_msg:
                    continue
                continue
    except Exception as e:
        print(f"Gemini Vision error: {e}")

    return fallback_image_analysis(image_path)


def fallback_image_analysis(image_path):
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
        if hasattr(gemini_client, 'models'):
            models_to_try = ['gemini-2.5-flash,gemini-1.5-flash']
            for model_name in models_to_try:
                try:
                    print(f"  🤖 Trying model: {model_name}")
                    response = gemini_client.models.generate_content(model=model_name, contents=prompt)
                    if response and response.text:
                        print(f"  ✅ Success with model: {model_name}")
                        return response.text
                except Exception as e:
                    error_msg = str(e)
                    print(f"  ✗ Model {model_name} error: {error_msg[:100]}")
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        print("  ⚠️ Rate limited, using fallback")
                        return get_enhanced_fallback_response(user_input, user_profile)
                    if "404" in error_msg or "NOT_FOUND" in error_msg:
                        continue
                    continue
        return get_enhanced_fallback_response(user_input, user_profile)
    except Exception as e:
        print(f"Gemini API error: {e}")
        return get_enhanced_fallback_response(user_input, user_profile)


def generate_default_tasks(plan):
    default_tasks = [
        {
            'title': 'Land Preparation',
            'description': 'Plowing, leveling, and adding manure',
            'due_date': plan.start_date - timedelta(days=7),
            'category': 'preparation'
        },
        {
            'title': 'Planting/Sowing',
            'description': f'Sow {plan.crop_type} seeds',
            'due_date': plan.start_date,
            'category': 'planting'
        },
        {
            'title': 'First Fertilizer Application',
            'description': 'Apply NPK fertilizer as per soil test',
            'due_date': plan.start_date + timedelta(days=30),
            'category': 'fertilizing'
        },
        {
            'title': 'Irrigation Check',
            'description': 'Ensure proper irrigation system',
            'due_date': plan.start_date + timedelta(days=15),
            'category': 'irrigation'
        },
        {
            'title': 'Pest Scouting',
            'description': 'Inspect for early signs of pests',
            'due_date': plan.start_date + timedelta(days=45),
            'category': 'pest_control'
        },
        {
            'title': 'Harvest',
            'description': 'Harvest mature crops',
            'due_date': plan.expected_harvest,
            'category': 'harvest'
        }
    ]
    for task_data in default_tasks:
        task = CropTask(
            plan_id=plan.id,
            title=task_data['title'],
            description=task_data['description'],
            due_date=task_data['due_date'],
            category=task_data['category']
        )
        db.session.add(task)
    db.session.commit()


# -------------------------------------------------------------------
# ADMIN DECORATOR
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Custom Jinja filters
# -------------------------------------------------------------------
@app.template_filter('fromjson')
def fromjson_filter(value):
    if not value or value == '[]':
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


@app.template_filter('timeago')
def timeago_filter(date):
    """Convert a datetime to a human-readable 'time ago' string."""
    if not date:
        return ''
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    diff = now - date
    if diff.days > 365:
        return f"{diff.days // 365} years ago"
    if diff.days > 30:
        return f"{diff.days // 30} months ago"
    if diff.days > 0:
        return f"{diff.days} days ago"
    if diff.seconds > 3600:
        return f"{diff.seconds // 3600} hours ago"
    if diff.seconds > 60:
        return f"{diff.seconds // 60} minutes ago"
    return "just now"


@app.template_filter('nl2br')
def nl2br_filter(s):
    if not s:
        return ''
    return s.replace('\n', '<br>\n')


# -------------------------------------------------------------------
# API and debug routes
# -------------------------------------------------------------------

@app.route('/api/list-gemini-models')
@login_required
def list_gemini_models():
    try:
        if not GEMINI_ENABLED or not gemini_client:
            return jsonify({"success": False, "message": "Gemini not enabled"})
        available_models = []
        test_models = ['gemini-2.5-flash,gemini-1.5-flash']
        for model_name in test_models:
            try:
                response = gemini_client.models.generate_content(model=model_name, contents="Say 'test'")
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
        return jsonify({"success": False, "error": str(e), "message": "Failed to list models"})


@app.route('/test-gemini')
def test_gemini():
    try:
        if not GEMINI_ENABLED or not gemini_client:
            return jsonify({
                "status": "disabled",
                "message": "Gemini is disabled or not configured",
                "gemini_enabled": GEMINI_ENABLED,
                "api_key_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY != "not_set")
            })
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
        return jsonify({"status": "error", "message": f"Failed to check API status: {str(e)[:100]}"})


@app.route('/test-key')
def test_key():
    return f"API Key is: {app.config.get('WEATHER_API_KEY', 'NOT FOUND')}"


# -------------------------------------------------------------------
# LOGIN MANAGER
# -------------------------------------------------------------------
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# -------------------------------------------------------------------
# PUBLIC ROUTES
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# AUTHENTICATION ROUTES
# -------------------------------------------------------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'GET':
        return render_template('register.html')
    if request.method == 'POST':
        try:
            email = request.form.get('email', '').strip().lower()
            name = request.form.get('name', '').strip()
            raw_phone = request.form.get('phone', '').strip()
            password = request.form.get('password', '').strip()
            confirm_password = request.form.get('confirm_password', '').strip()
            phone = re.sub(r'\D', '', raw_phone)
            if len(phone) >= 10:
                phone = f"+{phone}" if not phone.startswith('+') else phone
            else:
                phone = None

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

            existing_user = User.query.filter(db.func.lower(User.email) == email).first()
            if existing_user:
                flash('Email already registered. Please login instead.', 'danger')
                return render_template('register.html', form_data=request.form.to_dict())
            if phone and User.query.filter_by(phone=phone).first():
                flash('Phone number already registered.', 'danger')
                return render_template('register.html', form_data=request.form.to_dict())

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
                points_balance=100
            )
            user.set_password(password)

            optional_fields = [
                'whatsapp', 'gender', 'farm_name', 'secondary_crops',
                'soil_type', 'irrigation_type', 'district', 'farm_address',
                'preferred_language', 'referral_code'
            ]
            for field in optional_fields:
                value = request.form.get(field)
                if value:
                    setattr(user, field, value.strip())

            dob_str = request.form.get('dob')
            if dob_str:
                try:
                    user.dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
                except ValueError:
                    pass

            db.session.add(user)
            db.session.commit()

            activity = UserActivity(
                user_id=user.id,
                activity_type='registration',
                description='New user registered',
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string
            )
            db.session.add(activity)
            db.session.commit()

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
            user = User.query.filter(db.func.lower(User.email) == email).first()
            if user:
                if not user.is_active:
                    flash("Your account is not active. Please contact support.", "danger")
                    return render_template('login.html', form_data={'email': email})
                if user.check_password(password):
                    login_user(user, remember=True)
                    user.last_login = datetime.now(timezone.utc)
                    session['user_id'] = user.id
                    session['user_name'] = user.name
                    session['user_email'] = user.email
                    session['user_role'] = user.role
                    session['logged_in'] = True
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
                    next_page = request.args.get('next')
                    if next_page:
                        return redirect(next_page)
                    return redirect(url_for("dashboard"))
                else:
                    flash("Invalid email or password", "danger")
            else:
                flash(f"No account found with email: {email}. Please register first.", "warning")
                return render_template('login.html', form_data={'email': email}, suggest_register=True)
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
            activity = UserActivity(
                user_id=current_user.id,
                activity_type='logout',
                description='User logged out',
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string
            )
            db.session.add(activity)
            db.session.commit()
            session.clear()
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
    return render_template("profile.html", total_chats=total_chats, total_images=total_images)


# -------------------------------------------------------------------
# CHAT & AI ROUTES
# -------------------------------------------------------------------

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
        reply = get_local_response(message, user_profile)
        if not reply:
            if GEMINI_ENABLED:
                try:
                    gemini_reply = ask_gemini(message, user_profile)
                    if gemini_reply and len(gemini_reply.strip()) > 10:
                        reply = gemini_reply
                except Exception as e:
                    print(f"   ❌ Gemini error: {e}")
        if not reply or reply.strip() == "":
            reply = get_enhanced_fallback_response(message, user_profile)
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
            "source": "local_kb" if "local_kb" in str(locals()) else "gemini" if "gemini_reply" in locals() else "fallback"
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
        return jsonify({"status": "error", "message": f"Failed to check API status: {str(e)[:100]}"})


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
        history = [{
            "id": chat.id,
            "user_message": chat.user_message,
            "bot_response": chat.bot_response,
            "timestamp": chat.created_at.isoformat() if chat.created_at else None,
            "type": chat.chat_type
        } for chat in chats.items]
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


# -------------------------------------------------------------------
# IMAGE ANALYSIS
# -------------------------------------------------------------------

@app.route("/api/analyze-image", methods=["POST"])
@login_required
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400
        file = request.files['image']
        text_message = request.form.get('message', '').strip()
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
            return jsonify({"success": False, "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

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
        return jsonify({"success": False, "error": "Image analysis failed", "message": str(e)}), 500


@app.route("/api/image-analyses", methods=["GET"])
@login_required
def get_image_analyses():
    try:
        limit = request.args.get('limit', 10, type=int)
        analyses = ImageAnalysis.query.filter_by(
            user_id=current_user.id
        ).order_by(ImageAnalysis.created_at.desc()).limit(limit).all()
        result = [{
            "id": a.id,
            "filename": a.filename,
            "thumbnail": a.thumbnail,
            "user_message": a.user_message,
            "health_status": a.health_status,
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "confidence": a.confidence_score
        } for a in analyses]
        return jsonify({"success": True, "analyses": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/thumbnails/<filename>")
def uploaded_thumbnail(filename):
    return send_from_directory(app.config['THUMBNAIL_FOLDER'], filename)


# -------------------------------------------------------------------
# WEATHER ROUTES
# -------------------------------------------------------------------

logging.basicConfig(level=logging.DEBUG)


def generate_recommendations(weather):
    recs = []
    temp = weather.get('temperature', 28)
    humidity = weather.get('humidity', 65)
    wind_speed = float(weather.get('wind_speed', '12').split()[0])
    rain_prob = weather.get('hourly', [{}])[0].get('pop', 0) if weather.get('hourly') else 40

    if rain_prob > 60:
        recs.append(("Irrigation", "warning", "Rain expected – skip irrigation today."))
    elif temp > 35:
        recs.append(("Irrigation", "success", "High temperature – irrigate early morning."))
    else:
        recs.append(("Irrigation", "success", "Normal conditions – irrigate as usual."))

    if humidity > 80 and rain_prob > 40:
        recs.append(("Pest Alert", "warning", "High humidity + rain – fungal disease risk. Apply preventive spray."))
    else:
        recs.append(("Pest Alert", "info", "Low pest risk."))

    if rain_prob < 30 and wind_speed < 15:
        recs.append(("Fertilizer", "info", "Good conditions for foliar spray."))
    else:
        recs.append(("Fertilizer", "secondary", "Wait for calmer weather to apply fertilizer."))

    if rain_prob > 70:
        recs.append(("Harvest", "danger", "Heavy rain expected – harvest ripe crops immediately."))
    elif rain_prob > 30:
        recs.append(("Harvest", "warning", "Rain possible – consider harvesting if crops are ready."))
    else:
        recs.append(("Harvest", "success", "Ideal weather for harvesting."))

    return recs


@app.route('/weather')
@login_required
def weather():
    return render_template('weather.html')


@app.route('/api/weather', methods=['GET'])
@login_required
def get_weather():
    try:
        region = current_user.region or "Delhi"
        api_key = current_app.config.get('WEATHER_API_KEY')
        base_url = current_app.config.get('WEATHER_API_URL', 'https://api.openweathermap.org/data/2.5')

        if api_key and api_key != 'your_actual_key_here':
            try:
                current_params = {'q': region, 'appid': api_key, 'units': 'metric'}
                current_resp = requests.get(f"{base_url}/weather", params=current_params, timeout=10)
                current_resp.raise_for_status()
                current = current_resp.json()

                forecast_params = {'q': region, 'appid': api_key, 'units': 'metric'}
                forecast_resp = requests.get(f"{base_url}/forecast", params=forecast_params, timeout=10)
                forecast_resp.raise_for_status()
                forecast = forecast_resp.json()

                def format_time(timestamp):
                    return datetime.fromtimestamp(timestamp).strftime('%H:%M')

                weather_data = {
                    "temperature": round(current['main']['temp']),
                    "feels_like": round(current['main']['feels_like']),
                    "humidity": current['main']['humidity'],
                    "wind_speed": f"{round(current['wind']['speed'] * 3.6, 1)} km/h",
                    "conditions": current['weather'][0]['description'].title(),
                    "icon": current['weather'][0]['icon'],
                    "sunrise": format_time(current['sys']['sunrise']),
                    "sunset": format_time(current['sys']['sunset']),
                    "visibility": round(current.get('visibility', 10000) / 1000, 1),
                    "alerts": ["No severe weather alerts"],
                    "uvi": 5,
                    "hourly": [],
                    "daily": []
                }

                for item in forecast['list'][:8]:
                    dt = datetime.fromtimestamp(item['dt'])
                    weather_data['hourly'].append({
                        "time": dt.strftime('%I %p'),
                        "temp": round(item['main']['temp']),
                        "description": item['weather'][0]['description'].title(),
                        "icon": item['weather'][0]['icon'],
                        "pop": round(item['pop'] * 100)
                    })

                used_days = set()
                for item in forecast['list']:
                    day = datetime.fromtimestamp(item['dt']).strftime('%A')
                    if day not in used_days and len(weather_data['daily']) < 5:
                        used_days.add(day)
                        weather_data['daily'].append({
                            "day": day[:3],
                            "high": round(item['main']['temp_max']),
                            "low": round(item['main']['temp_min']),
                            "description": item['weather'][0]['description'].title(),
                            "icon": item['weather'][0]['icon'],
                            "pop": round(item['pop'] * 100)
                        })

                weather_data['recommendations'] = generate_recommendations(weather_data)
                return jsonify({"success": True, "weather": weather_data})
            except Exception as e:
                logging.error(f"Live API failed: {e}, using mock data")

        mock_data = {
            "temperature": 28,
            "feels_like": 30,
            "humidity": 65,
            "wind_speed": "12 km/h",
            "conditions": "Partly Cloudy",
            "icon": "02d",
            "sunrise": "06:15",
            "sunset": "18:45",
            "visibility": 10,
            "alerts": ["No severe weather alerts"],
            "uvi": 5,
            "hourly": [
                {"time": "06 AM", "temp": 22, "description": "Clear", "icon": "01d", "pop": 0},
                {"time": "09 AM", "temp": 25, "description": "Sunny", "icon": "01d", "pop": 0},
                {"time": "12 PM", "temp": 28, "description": "Partly Cloudy", "icon": "02d", "pop": 10},
                {"time": "03 PM", "temp": 29, "description": "Cloudy", "icon": "03d", "pop": 20},
                {"time": "06 PM", "temp": 26, "description": "Light Rain", "icon": "10d", "pop": 40},
                {"time": "09 PM", "temp": 23, "description": "Clear", "icon": "01n", "pop": 0},
            ],
            "daily": [
                {"day": "Mon", "high": 30, "low": 22, "description": "Sunny", "icon": "01d", "pop": 0},
                {"day": "Tue", "high": 29, "low": 23, "description": "Partly Cloudy", "icon": "02d", "pop": 10},
                {"day": "Wed", "high": 28, "low": 22, "description": "Cloudy", "icon": "03d", "pop": 20},
                {"day": "Thu", "high": 27, "low": 21, "description": "Rain", "icon": "10d", "pop": 60},
                {"day": "Fri", "high": 26, "low": 20, "description": "Thunderstorm", "icon": "11d", "pop": 80},
            ]
        }
        mock_data['recommendations'] = generate_recommendations(mock_data)
        return jsonify({"success": True, "weather": mock_data})
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/update_location', methods=['POST'])
@login_required
def update_location():
    data = request.get_json()
    new_city = data.get('city')
    if new_city:
        current_user.region = new_city
        db.session.commit()
        return jsonify(success=True, city=new_city)
    return jsonify(success=False, error="No city provided"), 400


# -------------------------------------------------------------------
# MARKET PRICES
# -------------------------------------------------------------------

@app.route('/market')
@login_required
def market_prices():
    return render_template('market.html')


from datetime import datetime, timedelta, timezone
from sqlalchemy import func

@app.route('/api/market-prices', methods=['GET'])
@login_required
def get_market_prices():
    try:
        # Get the latest prices (e.g., last 30 days, or most recent per crop)
        # For simplicity, we'll take the 20 most recent records
        prices = MarketPrice.query.order_by(MarketPrice.date.desc()).limit(20).all()

        price_list = []
        for p in prices:
            # Compute derived fields (replace with your own logic)
            # Example: min = 95% of price, max = 105% of price
            min_price = round(p.price * 0.95, 2)
            max_price = round(p.price * 1.05, 2)
            # Change: you'd need previous price; we'll use a placeholder
            change = "+2.5"  # Replace with real calculation
            # Demand: you can determine based on price or add a column
            demand = "High" if p.price > 50 else "Medium"  # Example
            # Category: you may need to add this to the model or infer
            category = "grains" if "wheat" in p.crop_name.lower() else "other"

            price_list.append({
                "crop": p.crop_name,
                "market": p.market_name,
                "min": min_price,
                "max": max_price,
                "avg": p.price,
                "change": change,
                "demand": demand,
                "category": category,
                "region": p.region,
                "date": p.date.isoformat() if p.date else None
            })

        # Quick stats – compute from your data (example)
        if prices:
            highest = max(prices, key=lambda x: x.price)
            stats = {
                "highest_price": {"crop": highest.crop_name, "price": highest.price},
                "most_volatile": {"crop": "Soybean", "change": -3.2},  # placeholder
                "best_buy": {"crop": "Corn", "change": 1.8},           # placeholder
                "demand_trend": {"crop": "Rice", "demand": "High"}     # placeholder
            }
        else:
            stats = {}

        # Trends – you can generate dummy or real data
        trends = {
            "labels": [f"Day {i+1}" for i in range(30)],
            "values": [2150 + i*10 for i in range(30)]  # dummy
        }

        return jsonify({
            "success": True,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "prices": price_list,
            "stats": stats,
            "trends": trends
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
# -------------------------------------------------------------------
# PEST DATABASE
# -------------------------------------------------------------------

@app.route('/pest-database')
@login_required
def pest_database():
    return render_template('pest_database.html')


@app.route('/api/pests', methods=['GET'])
@login_required
def get_pests():
    try:
        pests = [
            {"id": 1, "name": "Aphids", "crop_affected": "Tomato, Chilli, Cotton", "symptoms": "Curling leaves, stunted growth", "control": "Neem oil spray, Imidacloprid"},
            {"id": 2, "name": "Whiteflies", "crop_affected": "Tomato, Cotton, Soybean", "symptoms": "Yellowing leaves, sooty mold", "control": "Yellow sticky traps, Acetamiprid"},
            {"id": 3, "name": "Bollworms", "crop_affected": "Cotton, Chilli", "symptoms": "Holes in fruits/bolls", "control": "Bt cotton, Spinosad"}
        ]
        return jsonify({"success": True, "pests": pests})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# -------------------------------------------------------------------
# COMMUNITY FORUM
# -------------------------------------------------------------------

@app.route('/community')
@login_required
def community_forum():
    categories = ForumCategory.query.order_by(ForumCategory.name).all()
    recent_threads = ForumThread.query.order_by(
        ForumThread.is_pinned.desc(),
        ForumThread.updated_at.desc()
    ).limit(10).all()
    total_members = User.query.count()
    online_members = User.query.filter(User.last_login >= datetime.now(timezone.utc) - timedelta(minutes=15)).count()
    total_discussions = ForumThread.query.count()
    total_solutions = ForumPost.query.filter_by(is_solution=True).count()
    user_contributions = ForumThread.query.filter_by(user_id=current_user.id).count() + \
                         ForumPost.query.filter_by(user_id=current_user.id).count()
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    trending_tags = db.session.query(
        ForumTag.name,
        func.count(ForumThreadTag.thread_id).label('count')
    ).join(ForumThreadTag).join(ForumThread).filter(
        ForumThread.created_at >= week_ago
    ).group_by(ForumTag.id).order_by(desc('count')).limit(4).all()
    top_contributors = db.session.query(
        User,
        func.count(ForumPost.id).label('post_count')
    ).join(ForumPost).group_by(User.id).order_by(desc('post_count')).limit(5).all()
    return render_template('community.html',
                           categories=categories,
                           recent_threads=recent_threads,
                           total_members=total_members,
                           online_members=online_members,
                           total_discussions=total_discussions,
                           total_solutions=total_solutions,
                           user_contributions=user_contributions,
                           trending_tags=trending_tags,
                           top_contributors=top_contributors)


@app.route('/community/thread/<int:thread_id>')
@login_required
def view_thread(thread_id):
    thread = ForumThread.query.get_or_404(thread_id)
    thread.views += 1
    db.session.commit()
    return render_template('thread.html', thread=thread)


@app.route('/community/thread/new', methods=['POST'])
@login_required
def create_thread():
    title = request.form.get('title', '').strip()
    category_id = request.form.get('category_id')
    content = request.form.get('content', '').strip()
    tags = request.form.getlist('tags')
    if not title or not category_id or not content:
        flash('All fields are required.', 'danger')
        return redirect(url_for('community_forum'))
    thread = ForumThread(
        title=title,
        content=content,
        user_id=current_user.id,
        category_id=category_id
    )
    db.session.add(thread)
    db.session.flush()
    for tag_name in tags:
        tag = ForumTag.query.filter_by(name=tag_name).first()
        if not tag:
            tag = ForumTag(name=tag_name)
            db.session.add(tag)
            db.session.flush()
        thread_tag = ForumThreadTag(thread_id=thread.id, tag_id=tag.id)
        db.session.add(thread_tag)
    category = ForumCategory.query.get(category_id)
    if category:
        category.thread_count += 1
    db.session.commit()
    flash('Thread created successfully!', 'success')
    return redirect(url_for('view_thread', thread_id=thread.id))


@app.route('/community/thread/<int:thread_id>/reply', methods=['POST'])
@login_required
def post_reply(thread_id):
    thread = ForumThread.query.get_or_404(thread_id)
    if thread.is_locked:
        flash('This thread is locked.', 'warning')
        return redirect(url_for('view_thread', thread_id=thread_id))
    content = request.form.get('content', '').strip()
    if not content:
        flash('Reply cannot be empty.', 'danger')
        return redirect(url_for('view_thread', thread_id=thread_id))
    post = ForumPost(
        content=content,
        user_id=current_user.id,
        thread_id=thread_id
    )
    db.session.add(post)
    thread.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    flash('Reply posted.', 'success')
    return redirect(url_for('view_thread', thread_id=thread_id) + f'#post-{post.id}')


@app.route('/api/community/post/<int:post_id>/like', methods=['POST'])
@login_required
def like_post(post_id):
    post = ForumPost.query.get_or_404(post_id)
    like = ForumLike.query.filter_by(user_id=current_user.id, post_id=post_id).first()
    if like:
        db.session.delete(like)
        liked = False
    else:
        like = ForumLike(user_id=current_user.id, post_id=post_id)
        db.session.add(like)
        liked = True
    db.session.commit()
    return jsonify({'success': True, 'liked': liked, 'count': post.likes.count()})


@app.route('/community/search')
@login_required
def search_threads():
    q = request.args.get('q', '').strip()
    if not q:
        return redirect(url_for('community_forum'))

    threads = ForumThread.query.filter(
        (ForumThread.title.ilike(f'%{q}%')) |
        (ForumThread.content.ilike(f'%{q}%'))
    ).order_by(ForumThread.updated_at.desc()).all()

    return render_template('community_search.html', threads=threads, query=q)

@app.route('/debug/categories')
@login_required
def debug_categories():
    cats = ForumCategory.query.all()
    return jsonify([{'id': c.id, 'name': c.name} for c in cats])

@app.route('/community/category/<int:cat_id>')
@login_required
def category_view(cat_id):
    category = ForumCategory.query.get_or_404(cat_id)
    threads = ForumThread.query.filter_by(category_id=cat_id).order_by(
        ForumThread.is_pinned.desc(),
        ForumThread.updated_at.desc()
    ).all()
    return render_template('category_threads.html', category=category, threads=threads)
# -------------------------------------------------------------------
# COMMUNITY LIVE CHAT
# -------------------------------------------------------------------

@app.route('/chat-community')
@login_required
def chat_community():
    """General real-time chat room with initial messages"""
    messages_query = db.session.query(
        ChatMessage,
        User.name.label('sender_name'),
        User.role.label('sender_role'),
        User.profile_picture.label('sender_profile_picture')
    ).join(User, ChatMessage.sender_id == User.id) \
     .filter(ChatMessage.room == 'general') \
     .order_by(ChatMessage.created_at.desc()) \
     .limit(50).all()

    messages = []
    for msg, sender_name, sender_role, sender_pic in reversed(messages_query):
        # Get reactions for this message
        reactions = {}
        reaction_rows = db.session.query(
            MessageReaction.emoji,
            func.count(MessageReaction.id)
        ).filter(MessageReaction.message_id == msg.id) \
         .group_by(MessageReaction.emoji).all()
        for emoji, count in reaction_rows:
            reactions[emoji] = count

        # Get reply preview if any
        reply_to = None
        if msg.reply_to_id:
            reply_msg = ChatMessage.query.get(msg.reply_to_id)
            if reply_msg:
                reply_sender = User.query.get(reply_msg.sender_id)
                reply_to = {
                    'id': reply_msg.id,
                    'sender': reply_sender.name if reply_sender else 'Unknown',
                    'preview': (reply_msg.message[:50] + '...') if len(reply_msg.message) > 50 else reply_msg.message
                }

        messages.append({
            'id': msg.id,
            'sender_id': msg.sender_id,
            'sender': {
                'name': sender_name,
                'role': sender_role,
                'profile_picture': sender_pic
            },
            'message': msg.message,
            'file_url': msg.file_url,
            'file_type': msg.file_type,
            'reply_to': reply_to,
            'reactions': reactions,
            'created_at': msg.created_at,
            'read_by_all': False
        })

    return render_template('chat_community.html', messages=messages)


@app.route('/upload-chat-file', methods=['POST'])
@login_required
def upload_chat_file():
    """Handle file uploads for chat (images, voice, etc.)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = secure_filename(file.filename)
        unique_name = f"{current_user.id}_{int(time.time())}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)

        from mimetypes import guess_type
        mime_type, _ = guess_type(filename)
        if not mime_type:
            mime_type = 'application/octet-stream'

        file_url = url_for('uploaded_file', filename=unique_name, _external=True)

        return jsonify({'file_url': file_url, 'file_type': mime_type})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/follow/<int:user_id>', methods=['POST'])
@login_required
def follow_user(user_id):
    """Follow another user"""
    if user_id == current_user.id:
        return jsonify({'error': 'Cannot follow yourself'}), 400
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    existing = UserFollow.query.filter_by(follower_id=current_user.id, followed_id=user_id).first()
    if existing:
        return jsonify({'message': 'Already following'}), 200

    follow = UserFollow(follower_id=current_user.id, followed_id=user_id)
    db.session.add(follow)
    db.session.commit()
    return jsonify({'success': True, 'message': f'Now following {user.name}'})


@app.route('/unfollow/<int:user_id>', methods=['POST'])
@login_required
def unfollow_user(user_id):
    """Unfollow a user"""
    follow = UserFollow.query.filter_by(follower_id=current_user.id, followed_id=user_id).first()
    if follow:
        db.session.delete(follow)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Unfollowed'})
    return jsonify({'error': 'Not following'}), 404


@app.route('/user/<int:user_id>/profile')
@login_required
def user_profile_snippet(user_id):
    """Return HTML snippet for profile hover card"""
    user = db.session.get(User, user_id)
    if not user:
        return 'User not found', 404

    followers_count = UserFollow.query.filter_by(followed_id=user.id).count()
    following_count = UserFollow.query.filter_by(follower_id=user.id).count()
    is_following = UserFollow.query.filter_by(follower_id=current_user.id, followed_id=user.id).first() is not None

    html = f"""
    <div class="p-2" style="min-width: 200px;">
        <div class="d-flex align-items-center mb-2">
            <div class="avatar me-2" style="width: 48px; height: 48px;">
                {user.name[:2].upper()}
            </div>
            <div>
                <strong>{user.name}</strong><br>
                <small class="text-muted">@{user.email.split('@')[0]}</small>
            </div>
        </div>
        <p><i class="bi bi-geo-alt"></i> {user.region or 'Unknown'}</p>
        <p><i class="bi bi-tree"></i> {user.primary_crop or 'Not specified'}</p>
        <div class="d-flex justify-content-between">
            <span><strong>{followers_count}</strong> Followers</span>
            <span><strong>{following_count}</strong> Following</span>
        </div>
        {'<button class="btn btn-sm btn-primary mt-2 w-100" onclick="toggleFollow(' + str(user.id) + ')">Unfollow</button>' if is_following else '<button class="btn btn-sm btn-outline-primary mt-2 w-100" onclick="toggleFollow(' + str(user.id) + ')">Follow</button>'}
    </div>
    """
    return html


@app.route('/messages/send/<int:user_id>')
@login_required
def send_private_message(user_id):
    # Placeholder for private messaging
    flash('Private messaging coming soon!', 'info')
    return redirect(url_for('chat_community'))


# -------------------------------------------------------------------
# SOCKET.IO EVENT HANDLERS
# -------------------------------------------------------------------

# Online user tracking
online_users = {}      # room -> set of user ids
user_sid_map = {}      # sid -> user_id


@socketio.on('join')
def handle_join(data):
    room = data.get('room')
    username = data.get('username')
    user_id = current_user.id

    join_room(room)
    user_sid_map[request.sid] = user_id

    if room not in online_users:
        online_users[room] = set()
    online_users[room].add(user_id)

    emit('status', {'msg': f'{username} has joined the chat.'}, room=room, include_self=False)
    broadcast_online_users(room)


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    user_id = user_sid_map.get(sid)
    if user_id:
        for room, users in list(online_users.items()):
            if user_id in users:
                users.remove(user_id)
                broadcast_online_users(room)
        del user_sid_map[sid]


def broadcast_online_users(room):
    if room not in online_users:
        return
    user_list = []
    for uid in online_users[room]:
        user = User.query.get(uid)
        if user:
            user_list.append({
                'id': user.id,
                'name': user.name,
                'role': user.role
            })
    emit('user_count', {'count': len(user_list), 'users': user_list}, room=room)


@socketio.on('typing')
def handle_typing(data):
    room = data.get('room')
    sender = data.get('sender')
    sender_id = data.get('sender_id')
    emit('typing', {'sender': sender, 'sender_id': sender_id}, room=room, include_self=False)


@socketio.on('send_message')
def handle_send_message(data):
    room = data['room']
    msg_text = data.get('message', '')
    file_url = data.get('file_url')
    file_type = data.get('file_type')
    reply_to_id = data.get('reply_to_id')
    sender_id = data['sender_id']
    sender_name = data['sender']

    new_msg = ChatMessage(
        sender_id=sender_id,
        room=room,
        message=msg_text,
        file_url=file_url,
        file_type=file_type,
        reply_to_id=reply_to_id
    )
    db.session.add(new_msg)
    db.session.commit()

    sender = User.query.get(sender_id)
    sender_avatar = sender.profile_picture if sender else None

    reply_to = None
    if reply_to_id:
        reply_msg = ChatMessage.query.get(reply_to_id)
        if reply_msg:
            reply_sender = User.query.get(reply_msg.sender_id)
            reply_to = {
                'id': reply_msg.id,
                'sender': reply_sender.name if reply_sender else 'Unknown',
                'preview': (reply_msg.message[:50] + '...') if len(reply_msg.message) > 50 else reply_msg.message
            }

    timeago = datetime.now(timezone.utc).strftime('%H:%M')

    emit('message', {
        'id': new_msg.id,
        'sender': sender_name,
        'sender_id': sender_id,
        'sender_avatar': sender_avatar,
        'message': msg_text,
        'file_url': file_url,
        'file_type': file_type,
        'reply_to': reply_to,
        'timeago': timeago,
        'reactions': {}
    }, room=room)


@socketio.on('edit_message')
def handle_edit_message(data):
    msg_id = data['id']
    new_text = data['message']
    room = data['room']

    msg = ChatMessage.query.get(msg_id)
    if not msg or msg.sender_id != current_user.id:
        return

    msg.message = new_text
    db.session.commit()
    emit('message_edited', {'id': msg_id, 'message': new_text}, room=room)


@socketio.on('delete_message')
def handle_delete_message(data):
    msg_id = data['id']
    room = data['room']

    msg = ChatMessage.query.get(msg_id)
    if not msg:
        return
    if msg.sender_id != current_user.id and current_user.role not in ['admin', 'moderator']:
        return

    db.session.delete(msg)
    db.session.commit()
    emit('message_deleted', {'id': msg_id}, room=room)


@socketio.on('pin_message')
def handle_pin_message(data):
    if current_user.role not in ['admin', 'moderator']:
        return
    msg_id = data['id']
    room = data['room']
    emit('message_pinned', {'id': msg_id, 'message': 'Message pinned'}, room=room)


@socketio.on('reaction')
def handle_reaction(data):
    msg_id = data['message_id']
    emoji = data['emoji']
    room = data['room']

    reaction = MessageReaction.query.filter_by(
        message_id=msg_id,
        user_id=current_user.id,
        emoji=emoji
    ).first()

    if reaction:
        db.session.delete(reaction)
        added = False
    else:
        reaction = MessageReaction(
            message_id=msg_id,
            user_id=current_user.id,
            emoji=emoji
        )
        db.session.add(reaction)
        added = True
    db.session.commit()

    count = MessageReaction.query.filter_by(message_id=msg_id, emoji=emoji).count()

    emit('reaction_update', {
        'message_id': msg_id,
        'emoji': emoji,
        'count': count,
        'user_id': current_user.id,
        'added': added
    }, room=room)


@socketio.on('messages_seen')
def handle_messages_seen(data):
    room = data['room']
    last_seen_id = data['last_seen_id']
    emit('read_receipt', {
        'user_id': current_user.id,
        'user_name': current_user.name,
        'last_seen_id': last_seen_id
    }, room=room, include_self=False)


# -------------------------------------------------------------------
# DOCUMENTS
# -------------------------------------------------------------------

@app.route('/documents')
@login_required
def document_center():
    return render_template('docs.html')


@app.route('/documents/list/')
@login_required
def document_list():
    docs = Document.query.filter_by(user_id=current_user.id).order_by(Document.uploaded_at.desc()).all()
    data = [{
        'id': doc.id,
        'name': doc.name,
        'type': doc.file_type,
        'size': doc.formatted_size(),
        'date': doc.uploaded_at.strftime('%Y-%m-%d'),
        'category': doc.category,
        'description': doc.description,
        'url': doc.url,
    } for doc in docs]
    return jsonify(data)


@app.route('/documents/upload/', methods=['POST'])
@login_required
def document_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    files = request.files.getlist('file')
    uploaded = []
    for file in files:
        if file.filename == '':
            continue
        orig_filename = secure_filename(file.filename)
        name, ext = os.path.splitext(orig_filename)
        timestamp = int(time.time())
        filename = f"{name}_{timestamp}{ext}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER_DOCS'], filename)
        file.save(filepath)
        ext_lower = ext.lower().lstrip('.')
        if ext_lower in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
            file_type = 'image'
        elif ext_lower == 'pdf':
            file_type = 'pdf'
        elif ext_lower in ['doc', 'docx']:
            file_type = 'docs'
        elif ext_lower in ['xls', 'xlsx', 'csv']:
            file_type = 'spreadsheet'
        elif ext_lower in ['zip', 'rar', '7z']:
            file_type = 'archive'
        else:
            file_type = 'other'
        doc = Document(
            name=orig_filename,
            filename=filename,
            file_type=file_type,
            size=os.path.getsize(filepath),
            description=request.form.get('description', ''),
            category=request.form.get('category', 'uploads'),
            user_id=current_user.id
        )
        db.session.add(doc)
        db.session.commit()
        uploaded.append({
            'id': doc.id,
            'name': doc.name,
            'type': doc.file_type,
            'size': doc.formatted_size(),
            'date': doc.uploaded_at.strftime('%Y-%m-%d'),
            'category': doc.category,
            'description': doc.description,
            'url': doc.url,
        })
    return jsonify({'success': True, 'documents': uploaded})


@app.route('/documents/download/<int:doc_id>/')
@login_required
def document_download(doc_id):
    doc = Document.query.get_or_404(doc_id)
    if doc.user_id != current_user.id:
        abort(403)
    return send_file(doc.path, as_attachment=True, download_name=doc.name)


@app.route('/uploads/documents/<path:filename>')
def uploaded_document(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_DOCS'], filename)


# -------------------------------------------------------------------
# NOTIFICATIONS
# -------------------------------------------------------------------

@app.route('/notifications')
@login_required
def notifications():
    return render_template('notifications.html')


# -------------------------------------------------------------------
# CROP PLANNER
# -------------------------------------------------------------------

@app.route('/crop-planner')
@login_required
def crop_planner():
    today = datetime.now().date()
    if 3 <= today.month <= 6:
        season = "Rabi"
    elif 7 <= today.month <= 10:
        season = "Kharif"
    else:
        season = "Zaid"
    active_plans = CropPlan.query.filter_by(
        user_id=current_user.id, is_active=True
    ).order_by(CropPlan.start_date).all()
    for plan in active_plans:
        plan.days_since = (today - plan.start_date).days if plan.start_date else 0
        plan.days_to_harvest = (plan.expected_harvest - today).days if plan.expected_harvest else 0
    upcoming_tasks = CropTask.query.join(CropPlan).filter(
        CropPlan.user_id == current_user.id,
        CropPlan.is_active == True,
        CropTask.status == 'pending',
        CropTask.due_date >= today
    ).order_by(CropTask.due_date).limit(10).all()
    return render_template('crop_planner.html',
                           season=season,
                           active_plans=active_plans,
                           upcoming_tasks=upcoming_tasks,
                           current_user=current_user,
                           today=today)


@app.route('/api/crop-planner/create', methods=['POST'])
@login_required
def create_crop_plan():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        crop_type = data.get('crop_type')
        variety = data.get('variety')
        start_date_str = data.get('start_date')
        harvest_date_str = data.get('harvest_date')
        area = data.get('area')
        planting_method = data.get('planting_method')
        notes = data.get('notes')
        if not crop_type or not start_date_str or not harvest_date_str:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            harvest_date = datetime.strptime(harvest_date_str, '%Y-%m-%d').date()
            area = float(area) if area else None
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid date or area format'}), 400
        plan = CropPlan(
            user_id=current_user.id,
            crop_type=crop_type,
            variety=variety,
            start_date=start_date,
            expected_harvest=harvest_date,
            area=area,
            planting_method=planting_method,
            notes=notes,
            is_active=True
        )
        db.session.add(plan)
        db.session.commit()
        generate_default_tasks(plan)
        return jsonify({'success': True, 'plan_id': plan.id, 'message': 'Crop plan created successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/crop-planner/plans', methods=['GET'])
@login_required
def get_crop_plans():
    try:
        plans = CropPlan.query.filter_by(user_id=current_user.id, is_active=True).all()
        result = []
        for plan in plans:
            tasks = [{
                'id': t.id,
                'title': t.title,
                'due_date': t.due_date.isoformat() if t.due_date else None,
                'status': t.status,
                'category': t.category
            } for t in plan.tasks]
            result.append({
                'id': plan.id,
                'crop_type': plan.crop_type,
                'variety': plan.variety,
                'start_date': plan.start_date.isoformat() if plan.start_date else None,
                'expected_harvest': plan.expected_harvest.isoformat() if plan.expected_harvest else None,
                'area': plan.area,
                'planting_method': plan.planting_method,
                'notes': plan.notes,
                'tasks': tasks,
                'created_at': plan.created_at.isoformat() if plan.created_at else None
            })
        return jsonify({'success': True, 'plans': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/crop-planner/task/<int:task_id>/update', methods=['POST'])
@login_required
def update_task_status(task_id):
    try:
        task = CropTask.query.get_or_404(task_id)
        if task.plan.user_id != current_user.id:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        data = request.get_json() or {}
        new_status = data.get('status')
        if new_status in ['pending', 'completed', 'skipped']:
            task.status = new_status
            if new_status == 'completed':
                task.completed_at = datetime.now(timezone.utc)
            else:
                task.completed_at = None
            db.session.commit()
            return jsonify({'success': True, 'message': 'Task updated'})
        else:
            return jsonify({'success': False, 'error': 'Invalid status'}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/crop-planner/plan/<int:plan_id>/delete', methods=['DELETE'])
@login_required
def delete_crop_plan(plan_id):
    try:
        plan = CropPlan.query.get_or_404(plan_id)
        if plan.user_id != current_user.id:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        db.session.delete(plan)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Plan deleted'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


# -------------------------------------------------------------------
# FARMING TIPS
# -------------------------------------------------------------------

@app.route('/api/farming-tips', methods=['GET'])
@login_required
def get_farming_tips():
    try:
        tips = FarmingTip.query.filter_by(is_active=True).order_by(FarmingTip.created_at.desc()).limit(10).all()
        tip_list = [{
            "id": t.id,
            "title": t.title,
            "content": t.content,
            "category": t.category,
            "crop_type": t.crop_type,
            "region": t.region,
            "language": t.language,
            "created_at": t.created_at.isoformat() if t.created_at else None
        } for t in tips]
        return jsonify({"success": True, "tips": tip_list})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# -------------------------------------------------------------------
# ADMIN ROUTES
# -------------------------------------------------------------------

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
        new_users_week = User.query.filter(User.created_at >= week_ago).count()
        recent_users = User.query.order_by(User.id.desc()).limit(10).all()
        recent_chats = ChatHistory.query.order_by(ChatHistory.created_at.desc()).limit(10).all()
        active_users = db.session.query(
            User, func.count(ChatHistory.id).label('chat_count')
        ).join(ChatHistory).group_by(User.id).order_by(desc('chat_count')).limit(10).all()
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
        # User growth over time
        user_growth_data = db.session.query(
            func.date(User.created_at).label('date'),
            func.count(User.id).label('count')
        ).group_by(func.date(User.created_at)).order_by(func.date(User.created_at)).all()
        user_growth_labels = [str(row.date) for row in user_growth_data]  # convert date to string
        user_growth_counts = [row.count for row in user_growth_data]

        # Chat activity over time
        chat_activity_data = db.session.query(
            func.date(ChatHistory.created_at).label('date'),
            func.count(ChatHistory.id).label('count')
        ).group_by(func.date(ChatHistory.created_at)).order_by(func.date(ChatHistory.created_at)).all()
        chat_activity_labels = [str(row.date) for row in chat_activity_data]
        chat_activity_counts = [row.count for row in chat_activity_data]

        # Top crops
        top_crops_data = db.session.query(
            User.primary_crop,
            func.count(User.id).label('user_count')
        ).filter(User.primary_crop.isnot(None), User.primary_crop != '').group_by(User.primary_crop).order_by(desc('user_count')).limit(10).all()
        top_crops_labels = [row[0] or 'Unknown' for row in top_crops_data]
        top_crops_counts = [row[1] for row in top_crops_data]

        # Other metrics
        total_users = User.query.count()
        active_users = db.session.query(
            func.count(func.distinct(ChatHistory.user_id))
        ).filter(ChatHistory.created_at >= datetime.now() - timedelta(days=30)).scalar() or 0
        avg_chats_per_user = db.session.query(
            func.avg(
                db.session.query(
                    func.count(ChatHistory.id)
                ).filter(ChatHistory.created_at >= datetime.now() - timedelta(days=30)).group_by(ChatHistory.user_id).subquery().c.count
            )
        ).scalar() or 0
        total_chats = ChatHistory.query.count()
        total_images = ImageAnalysis.query.count()

        return render_template(
            "admin/analytics.html",
            # Original data (for tables or fallback)
            user_growth=user_growth_data,
            chat_activity=chat_activity_data,
            top_crops=top_crops_data,
            # JSON-encoded data for charts
            user_growth_labels=json.dumps(user_growth_labels),
            user_growth_counts=json.dumps(user_growth_counts),
            chat_activity_labels=json.dumps(chat_activity_labels),
            chat_activity_counts=json.dumps(chat_activity_counts),
            top_crops_labels=json.dumps(top_crops_labels),
            top_crops_counts=json.dumps(top_crops_counts),
            # Metrics
            total_users=total_users,
            active_users=active_users,
            avg_chats_per_user=round(avg_chats_per_user, 2),
            total_chats=total_chats,
            total_images=total_images
        )
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
        writer.writerow(['ID', 'Email', 'Name', 'Phone', 'Role', 'Region', 'Farm Size',
                         'Primary Crop', 'Experience Level', 'Status', 'Created At'])
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
        writer.writerow(['ID', 'User ID', 'User Message', 'Bot Response', 'Type', 'Language', 'Created At'])
        for chat in chats:
            writer.writerow([
                chat.id,
                chat.user_id,
                (chat.user_message or '')[:500],
                (chat.bot_response or '')[:500],
                chat.chat_type or '',
                chat.language or '',
                chat.created_at.strftime('%Y-%m-%d %H:%M:%S') if chat.created_at else ''
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
                               market_prices=market_prices,
                               today=date.today().isoformat())
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

@app.route("/admin/farming-tips/edit/<int:tip_id>", methods=["GET", "POST"])
@login_required
@admin_required
def admin_edit_farming_tip(tip_id):
    tip = FarmingTip.query.get_or_404(tip_id)
    if request.method == "POST":
        try:
            tip.title = request.form.get("title", "").strip()
            tip.content = request.form.get("content", "").strip()
            tip.category = request.form.get("category", "general")
            tip.crop_type = request.form.get("crop_type")
            tip.language = request.form.get("language", "en")
            if not tip.title or not tip.content:
                flash("Title and content are required", "danger")
                return redirect(url_for('admin_edit_farming_tip', tip_id=tip_id))
            db.session.commit()
            flash("Farming tip updated successfully", "success")
            return redirect(url_for('admin_knowledge_base'))
        except Exception as e:
            db.session.rollback()
            flash(f"Error updating farming tip: {str(e)}", "danger")
            return redirect(url_for('admin_edit_farming_tip', tip_id=tip_id))
    # GET request: show edit form
    return render_template("admin/edit_farming_tip.html", tip=tip)


@app.route("/admin/farming-tips/delete/<int:tip_id>", methods=["POST"])
@login_required
@admin_required
def admin_delete_farming_tip(tip_id):
    try:
        tip = FarmingTip.query.get_or_404(tip_id)
        db.session.delete(tip)
        db.session.commit()
        flash("Farming tip deleted successfully", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting farming tip: {str(e)}", "danger")
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
            source=source,
            date=date.today()   # <-- ADD THIS LINE
        )
        db.session.add(price_entry)
        db.session.commit()
        flash("Market price added successfully", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error adding market price: {str(e)}", "danger")
    return redirect(url_for('admin_knowledge_base'))

@app.route("/admin/market-prices/edit/<int:price_id>", methods=["GET", "POST"])
@login_required
@admin_required
def admin_edit_market_price(price_id):
    price = MarketPrice.query.get_or_404(price_id)
    if request.method == "POST":
        try:
            price.crop_name = request.form.get("crop_name", "").strip()
            price.market_name = request.form.get("market_name", "").strip()
            price.region = request.form.get("region", "").strip()
            price.price = request.form.get("price", type=float)
            price.unit = request.form.get("unit", "kg")
            price.source = request.form.get("source", "")
            date_str = request.form.get("date")
            if not price.crop_name or not price.price:
                flash("Crop name and price are required", "danger")
                return redirect(url_for('admin_edit_market_price', price_id=price_id))
            if date_str:
                price.date = datetime.strptime(date_str, '%Y-%m-%d').date()
            db.session.commit()
            flash("Market price updated successfully", "success")
            return redirect(url_for('admin_knowledge_base'))
        except Exception as e:
            db.session.rollback()
            flash(f"Error updating market price: {str(e)}", "danger")
            return redirect(url_for('admin_edit_market_price', price_id=price_id))
    # GET request: show edit form
    return render_template("admin/edit_market_price.html", price=price)


@app.route("/admin/market-prices/delete/<int:price_id>", methods=["POST"])
@login_required
@admin_required
def admin_delete_market_price(price_id):
    try:
        price = MarketPrice.query.get_or_404(price_id)
        db.session.delete(price)
        db.session.commit()
        flash("Market price deleted successfully", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting market price: {str(e)}", "danger")
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
        # Database connection test
        db_status = "Healthy"
        try:
            db.session.execute(text("SELECT 1"))
        except Exception as e:
            db_status = f"Error: {str(e)}"

        # File system checks
        upload_folder_exists = os.path.exists(app.config['UPLOAD_FOLDER'])
        upload_folder_writable = os.access(app.config['UPLOAD_FOLDER'], os.W_OK)

        # Gemini API status – create a simple string and booleans
        gemini_configured = bool(GEMINI_API_KEY and GEMINI_API_KEY != "not_set")
        gemini_working = GEMINI_ENABLED
        if gemini_working:
            gemini_status_str = "Working"
        elif not gemini_configured:
            gemini_status_str = "Not Configured"
        else:
            gemini_status_str = "Error"

        # Disk usage
        total, used, free = shutil.disk_usage("/")
        disk_usage = {
            'total_gb': round(total / (1024 ** 3), 2),
            'used_gb': round(used / (1024 ** 3), 2),
            'free_gb': round(free / (1024 ** 3), 2),
            'percent_used': round((used / total) * 100, 2)
        }

        # Database counts
        user_count = User.query.count()
        chat_count = ChatHistory.query.count()
        image_count = ImageAnalysis.query.count()
        tip_count = FarmingTip.query.count()
        profile_pics_count = User.query.filter(User.profile_picture.isnot(None)).count()

        # Upload folder path (as string)
        upload_folder_path = app.config['UPLOAD_FOLDER']

        # Current time
        now = datetime.now()

        return render_template(
            "admin/system_health.html",
            db_status=db_status,
            gemini_status=gemini_status_str,          # now a simple string
            gemini_configured=gemini_configured,       # boolean for key status
            upload_folder_exists=upload_folder_exists,
            upload_folder_writable=upload_folder_writable,
            upload_folder_path=upload_folder_path,
            disk_usage=disk_usage,
            user_count=user_count,
            chat_count=chat_count,
            image_count=image_count,
            tip_count=tip_count,
            profile_pics_count=profile_pics_count,
            now=now,                                    # datetime object for last checked
            GEMINI_API_KEY=GEMINI_API_KEY               # only if you need key length (optional)
        )
    except Exception as e:
        flash(f"Error checking system health: {str(e)}", "danger")
        return redirect(url_for('admin_dashboard'))

@app.route("/admin/delete-chat/<int:chat_id>", methods=["POST"])
@login_required
@admin_required
def admin_delete_chat(chat_id):
    try:
        chat = ChatHistory.query.get_or_404(chat_id)
        db.session.delete(chat)
        db.session.commit()
        flash("Chat deleted successfully", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting chat: {str(e)}", "danger")
    return redirect(request.referrer or url_for('admin_chats'))


# -------------------------------------------------------------------
# UTILITY ROUTES
# -------------------------------------------------------------------

@app.route('/api/crop-schedule', methods=['POST'])
@login_required
def crop_schedule():
    try:
        data = request.get_json() or {}
        crop = data.get('crop', '')
        region = current_user.region or 'India'
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
    try:
        db.create_all()
        admin_user = User.query.filter_by(email='admin@aiagrobot.com').first()
        if not admin_user:
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
                FarmingTip(title="Watering Best Practices", content="Water your crops early in the morning to reduce evaporation loss.", category="general", language="en"),
                FarmingTip(title="Organic Pest Control", content="Use neem oil spray (2ml per liter of water) to control common pests.", category="general", language="en"),
            ]
            for tip in sample_tips:
                db.session.add(tip)
            sample_prices = [
                MarketPrice(crop_name="Rice", market_name="Mandi Bhav", region="Punjab", price=28.50, unit="kg", date=datetime.now().date(), source="Government Portal"),
            ]
            for price in sample_prices:
                db.session.add(price)
            db.session.commit()
            return jsonify({"success": True, "message": "Database initialized successfully"})
        else:
            return jsonify({"success": True, "message": "Database already exists"})
    except Exception as e:
        db.session.rollback()
        print(f"❌ Database initialization error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/backup-db")
@login_required
@admin_required
def backup_database():
    try:
        if app.config['SQLALCHEMY_DATABASE_URI'].startswith('sqlite'):
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
        return jsonify({"success": False, "message": "Database backup not supported for this database type"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/test-users")
def test_users():
    users = User.query.all()
    result = [{
        'id': u.id,
        'email': u.email,
        'name': u.name,
        'role': u.role,
        'is_active': u.is_active,
        'created_at': u.created_at.isoformat() if u.created_at else None
    } for u in users]
    return jsonify({"users": result, "count": len(result)})


@app.route("/check-db")
def check_database():
    try:
        with app.app_context():
            db_info = {
                'database_uri': app.config['SQLALCHEMY_DATABASE_URI'],
                'database_exists': os.path.exists(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')) if 'sqlite' in app.config['SQLALCHEMY_DATABASE_URI'] else 'N/A (PostgreSQL)',
                'total_users': User.query.count(),
                'users': [{'id': u.id, 'email': u.email, 'name': u.name, 'created_at': u.created_at.isoformat() if u.created_at else None} for u in User.query.all()]
            }
            return jsonify({'success': True, 'database_info': db_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


@app.errorhandler(404)
def not_found_error(error):
    return "Page not found", 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return "Internal server error", 500


@app.route('/static/<path:filename>')
def static_files(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except:
        return "File not found", 404


# -------------------------------------------------------------------
# CREATE DEFAULT FORUM CATEGORIES (run once)
# -------------------------------------------------------------------
def create_default_forum_categories():
    default_categories = [
        {'name': 'Crop Cultivation', 'description': 'Discuss growing techniques, varieties, and best practices',
         'icon': 'bi-tree-fill', 'color': 'success', 'thread_count': 0},
        {'name': 'Pest Control', 'description': 'Share pest management solutions and experiences',
         'icon': 'bi-bug-fill', 'color': 'warning', 'thread_count': 0},
        {'name': 'Market & Sales', 'description': 'Discuss prices, marketing, and selling strategies',
         'icon': 'bi-cash-coin', 'color': 'info', 'thread_count': 0},
        {'name': 'Equipment', 'description': 'Discuss farm machinery, tools, and maintenance',
         'icon': 'bi-tools', 'color': 'primary', 'thread_count': 0},
        {'name': 'Weather', 'description': 'Share weather updates and farming adaptations',
         'icon': 'bi-cloud-sun-fill', 'color': 'danger', 'thread_count': 0},
        {'name': 'Q&A', 'description': 'Ask questions and get answers from experts',
         'icon': 'bi-question-circle-fill', 'color': 'secondary', 'thread_count': 0},
    ]
    for cat_data in default_categories:
        existing = ForumCategory.query.filter_by(name=cat_data['name']).first()
        if not existing:
            db.session.add(ForumCategory(**cat_data))
    db.session.commit()
    print("✅ Default forum categories inserted.")


# -------------------------------------------------------------------
# DATABASE INITIALIZATION
# -------------------------------------------------------------------
def init_database():
    with app.app_context():
        try:
            inspector = inspect(db.engine)
            existing_tables = inspector.get_table_names()
            print(f"📊 Existing tables: {existing_tables}")
            db.create_all()
            create_default_forum_categories()
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


# -------------------------------------------------------------------
# CREATE MISSING FILES
# -------------------------------------------------------------------
def create_missing_files():
    templates_dir = os.path.join(basedir, "templates")
    os.makedirs(templates_dir, exist_ok=True)
    admin_templates_dir = os.path.join(templates_dir, "admin")
    os.makedirs(admin_templates_dir, exist_ok=True)
    static_dirs = ['css', 'js', 'images', 'uploads', 'thumbnails']
    for dir_name in static_dirs:
        os.makedirs(os.path.join(basedir, "static", dir_name), exist_ok=True)
    css_path = os.path.join(basedir, "static", "css", "style.css")
    if not os.path.exists(css_path):
        with open(css_path, 'w') as f:
            f.write("""/* Default CSS for AI-AgroBot */
body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
.header { background-color: #4CAF50; color: white; padding: 20px; text-align: center; }
.navbar { background-color: #333; overflow: hidden; }
.navbar a { float: left; color: white; text-align: center; padding: 14px 16px; text-decoration: none; }
.navbar a:hover { background-color: #ddd; color: black; }
.btn { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; }
.btn:hover { background-color: #45a049; }
.form-group { margin-bottom: 15px; }
.form-control { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; }
.alert { padding: 10px; margin: 10px 0; border-radius: 5px; }
.alert-success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
.alert-danger { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
.alert-info { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
.alert-warning { background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
.card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.table { width: 100%; border-collapse: collapse; }
.table th, .table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
.table th { background-color: #f2f2f2; font-weight: bold; }
.chat-container { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.chat-message { margin: 10px 0; padding: 10px; border-radius: 5px; }
.user-message { background-color: #e3f2fd; text-align: right; }
.bot-message { background-color: #f5f5f5; text-align: left; }
.dashboard-stats { display: flex; justify-content: space-between; flex-wrap: wrap; margin: 20px 0; }
.stat-card { background: white; border-radius: 8px; padding: 20px; margin: 10px; flex: 1; min-width: 200px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.stat-number { font-size: 2em; font-weight: bold; color: #4CAF50; }
.stat-label { color: #666; margin-top: 5px; }
""")
    image_path = os.path.join(basedir, "static", "images", "farmer-ai.svg")
    if not os.path.exists(image_path):
        with open(image_path, 'w') as f:
            f.write('<svg width="400" height="300" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#4CAF50" opacity="0.1"/><circle cx="200" cy="150" r="80" fill="#4CAF50" opacity="0.3"/><path d="M150,100 L250,100 L200,200 Z" fill="#4CAF50" opacity="0.5"/><text x="200" y="280" text-anchor="middle" font-family="Arial" font-size="20" fill="#333">AI AgroBot</text></svg>')
    print("✅ Created missing static files")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("🌾 AI-AgroBot Server Starting...")
    print("=" * 50)
    create_missing_files()
    init_database()
    print("\n✅ Server is running!")
    print("🌐 Open: http://localhost:5000")
    print("🔑 Default admin: admin@aiagrobot.com / admin123")
    print("🔑 Demo farmer: demo@aiagrobot.com / demo123")
    print(f"📁 Database file: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"🤖 Gemini AI: {'✅ Enabled' if GEMINI_ENABLED else '❌ Disabled'}")
    print("🔍 Check database: http://localhost:5000/check-db")
    print("🔍 Test Gemini: http://localhost:5000/test-gemini")
    print("=" * 50)
    socketio.run(app, debug=True, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
