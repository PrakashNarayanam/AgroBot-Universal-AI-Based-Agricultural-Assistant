# AgroBot – AI-Powered Agricultural Assistant
## Complete Project Documentation

---

## 1. Title

### **AgroBot – Smart AI Agricultural Assistant for Indian Farmers**

> An intelligent, full-stack web application that empowers Indian farmers with real-time AI-driven crop advice, disease detection, weather forecasting, and community collaboration.

---

## 2. Project Statement

### Problem Statement

Indian agriculture faces critical challenges:
- **Lack of timely expert advice**: Farmers in remote areas have limited access to agricultural experts.
- **Crop disease & pest losses**: Early identification of plant diseases can prevent up to 40% crop losses.
- **Market price opacity**: Farmers often sell at below-market rates due to lack of pricing information.
- **Language & digital barriers**: Most platforms are not farmer-friendly or region-specific.
- **Poor planning tools**: Absence of structured crop planning leads to inefficient resource usage.

### Solution – AgroBot

AgroBot is an AI-powered agricultural assistant platform that provides:
1. A conversational AI chatbot (powered by Google Gemini) for farming queries.
2. Image-based plant disease and health detection using AI vision.
3. Real-time weather data with farming-specific recommendations.
4. Live crop market prices from across India.
5. A community forum for farmer collaboration.
6. Crop planning and task management tools.
7. A comprehensive pest and disease knowledge database.

---

## 3. Outcomes

### Expected Outcomes

| Outcome | Description |
|---|---|
| **Increased Crop Yield** | Timely advice on fertilizers, irrigation, and pest control improves yield by 20–40%. |
| **Early Disease Detection** | AI image analysis detects diseases at early stages, reducing crop loss. |
| **Informed Market Decisions** | Real-time price visibility helps farmers get fair prices. |
| **Farmer Engagement** | Community chat and forums create a collaborative farming network. |
| **Better Planning** | Crop planner helps organize tasks from sowing to harvest. |
| **Accessibility** | Multi-language support and simple UI makes it usable for all farmers. |
| **Data-Driven Insights** | Admin dashboard tracks usage patterns and agricultural trends. |

### Success Metrics
- Number of registered farmers
- AI chat interactions per day
- Images analyzed per week
- Community engagement rate
- User satisfaction score

---

## 4. Technology

### Tech Stack

#### Backend
| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.12+ | Core programming language |
| **Flask** | 3.1.2 | Web framework |
| **Flask-SQLAlchemy** | 3.1.1 | ORM for database operations |
| **Flask-Login** | 0.6.3 | User authentication & session management |
| **Flask-SocketIO** | 5.6.1 | Real-time WebSocket communication |
| **SQLAlchemy** | 2.0.46 | Database ORM |
| **Werkzeug** | Latest | Password hashing, file handling |

#### AI / Machine Learning
| Technology | Purpose |
|---|---|
| **Google Gemini API** (google-genai 1.66.0) | AI chatbot (text) and vision (image analysis) |
| **Gemini 2.0 Flash / 1.5 Flash** | Model used for crop advice and disease detection |
| **PIL (Pillow)** | Fallback image processing (green pixel ratio analysis) |

#### Frontend
| Technology | Purpose |
|---|---|
| **HTML5 / CSS3** | Structure and styling |
| **Bootstrap 5 / Custom CSS** | Responsive UI design |
| **JavaScript (Vanilla + jQuery)** | Interactive UI, AJAX API calls |
| **Socket.IO (Client)** | Real-time chat |
| **Jinja2** | Server-side templating engine |

#### Database
| Database | Use Case |
|---|---|
| **SQLite** (`agrobot.db`) | Development / Local database |
| **PostgreSQL** | Production database (configurable via `DATABASE_URL`) |

#### External APIs
| API | Purpose |
|---|---|
| **OpenWeatherMap API** | Live weather data and 5-day forecast |
| **Google Gemini API** | AI-powered text and image analysis |

#### DevOps / Environment
| Tool | Purpose |
|---|---|
| **python-dotenv** | Environment variable management |
| **uv / pip** | Dependency management |
| **PyProject.toml** | Project configuration |

---

## 5. Modules to be Implemented

### Module 1: User Authentication & Profile Management
- **Register/Login/Logout** with email + password
- **Profile Management**: Farm name, size, region, crop types, experience level
- **Role-based access**: Farmer role / Admin role
- **Activity Logging**: Track login, logout, and other activities
- **Points System**: Reward system for user engagement

### Module 2: AI Chatbot (Text)
- **Conversational AI** using Google Gemini 2.0 Flash
- **Contextual Responses**: User profile (region, crop type) is passed as context
- **Local Knowledge Base**: Pre-loaded crop knowledge for Rice, Wheat, Maize, Pests, Soil
- **Chat History**: Stores all conversations in the database
- **Fallback System**: Graceful degradation when AI is unavailable

### Module 3: Image Analysis (AI Vision)
- **Plant Disease Detection** via Gemini Vision API
- **Upload & Analyze**: Supports JPG, PNG, GIF, BMP, WEBP formats
- **Thumbnail Generation**: Auto-generates image thumbnails using Pillow
- **Pixel-Based Fallback**: If AI is unavailable, analyzes green pixel ratio for plant health
- **Analysis History**: Stores all image analyses with confidence scores

### Module 4: Weather Forecasting
- **Live Weather**: Integrates with OpenWeatherMap API
- **5-Day Forecast**: Daily and hourly forecasts
- **Farming Recommendations**: Auto-generates irrigation, fertilizer, and harvest recommendations based on weather
- **Location Update**: Farmers can change their region dynamically
- **Mock Data Fallback**: Uses mock data if API is unavailable

### Module 5: Market Price Intelligence
- **Live Crop Prices**: Display current market prices for agricultural commodities
- **Price Trends**: 30-day historical chart visualization
- **Statistics**: Highest price, most volatile, best buy crop indicators
- **Demand Indicators**: High/Medium difficulty demand classification
- **Region-wise Prices**: Filter by region / market

### Module 6: Pest & Disease Database
- **Pest Encyclopedia**: Searchable database of agricultural pests
- **Crop-wise Filtering**: Filter pests by crop type
- **Symptoms & Controls**: Detailed information on symptoms and treatment methods
- **Visual Gallery**: Images of pests and affected crops

### Module 7: Crop Planner
- **Crop Plan Creation**: Set crop type, variety, planting date, harvest date, area
- **Task Management**: Auto-generates farming tasks (land prep, planting, fertilizing, etc.)
- **Task Tracking**: Mark tasks as pending/completed
- **Calendar View**: Visual timeline of crop activities

### Module 8: Community Forum
- **Categorized Threads**: Topics like soil health, pest control, market prices, etc.
- **Thread Creation & Replies**: Full forum functionality
- **Post Liking**: Engagement mechanism
- **Tagging System**: Searchable tags on threads
- **Trending Tags**: Shows most discussed topics of the week
- **Top Contributors**: Leaderboard for most active members

### Module 9: Real-Time Community Chat
- **WebSocket Chat** via Flask-SocketIO
- **General Chat Room**: All farmers chat in real time
- **File Sharing**: Share images in chat
- **Message Reactions**: Emoji reactions on messages
- **Reply Threading**: Reply to specific messages
- **Private Messaging**: Direct messages between users

### Module 10: Admin Dashboard
- **User Management**: View, activate/deactivate users
- **Analytics**: Total users, chats, image analyses
- **Activity Monitoring**: Track user activities
- **Database Management**: SQLite / PostgreSQL support

### Module 11: Document Management
- **Upload Documents**: PDF, TXT, DOC, DOCX, CSV support
- **Categorized Storage**: Organize farming documents
- **Download & Share**: Access documents anytime

### Module 12: Notifications
- **Weather Alerts**: Severe weather notifications
- **Pest Alerts**: When pest risk is high
- **Market Alerts**: Price fluctuation notifications
- **Customizable**: Users can toggle notification preferences

---

## 6. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│         Browser (HTML/CSS/JS + Socket.IO Client)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP / WebSocket
┌──────────────────────────▼──────────────────────────────────────┐
│                      FLASK WEB SERVER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Auth       │  │   API        │  │   SocketIO           │  │
│  │   Routes     │  │   Routes     │  │   (Real-time Chat)   │  │
│  │ /login       │  │ /api/chat    │  │   join_room()        │  │
│  │ /register    │  │ /api/weather │  │   emit()             │  │
│  │ /logout      │  │ /api/market  │  └──────────────────────┘  │
│  └──────────────┘  └──────────────┘                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              BUSINESS LOGIC LAYER                        │  │
│  │  ask_gemini() | analyze_with_gemini() | get_weather()    │  │
│  │  get_local_response() | fallback_image_analysis()        │  │
│  │  generate_recommendations() | generate_default_tasks()   │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────┬──────────────────────────┬────────────────────────┘
             │                          │
┌────────────▼───────┐    ┌─────────────▼──────────────────────┐
│   SQLAlchemy ORM   │    │       EXTERNAL APIs                │
│   Database Layer   │    │  ┌──────────────────────────────┐  │
│                    │    │  │  Google Gemini API            │  │
│  SQLite / Postgres │    │  │  (Text + Vision AI)           │  │
│                    │    │  └──────────────────────────────┘  │
│  Tables:           │    │  ┌──────────────────────────────┐  │
│  - users           │    │  │  OpenWeatherMap API           │  │
│  - chat_history    │    │  │  (Weather Forecasting)        │  │
│  - image_analyses  │    │  └──────────────────────────────┘  │
│  - market_prices   │    └────────────────────────────────────┘
│  - crop_plans      │
│  - forum_threads   │
│  - chat_messages   │
│  - documents       │
│  - weather_alerts  │
│  (+ 10 more...)    │
└────────────────────┘
```

### Request Flow
```
User Query → Flask Route → Auth Check → Business Logic
    → Gemini AI / Local KB / Weather API → Response
    → Database Save → JSON Response → Browser Render
```

---

## 7. Database Schema

### Core Tables

#### `users`
| Column | Type | Description |
|---|---|---|
| id | Integer (PK) | Unique user identifier |
| email | String(120) | Unique email address |
| password | String(200) | Hashed password |
| name | String(100) | Full name |
| phone | String(20) | Contact number |
| farm_name | String(100) | Name of the farm |
| farm_size | String(100) | Farm size (acres) |
| primary_crop | String(100) | Main crop grown |
| secondary_crops | Text | Other crops |
| soil_type | String(50) | Soil classification |
| irrigation_type | String(50) | Irrigation method |
| region | String(100) | Geographic region |
| experience_level | String(50) | beginner/intermediate/expert |
| preferred_language | String(10) | Language code |
| role | String(20) | farmer / admin |
| points_balance | Integer | Reward points |
| is_active | Boolean | Account status |
| created_at | DateTime | Registration timestamp |

#### `chat_history`
| Column | Type | Description |
|---|---|---|
| id | Integer (PK) | Unique ID |
| user_id | FK → users | Owner |
| user_message | Text | User's query |
| bot_response | Text | AI response |
| chat_type | String(20) | text / image |
| image_filename | String(200) | If image chat |
| language | String(10) | Response language |

#### `image_analyses`
| Column | Type | Description |
|---|---|---|
| id | Integer (PK) | Unique ID |
| user_id | FK → users | Owner |
| filename | String(200) | Uploaded image file |
| thumbnail | String(200) | Thumbnail filename |
| health_status | String(100) | Plant health result |
| analysis_result | Text | Full AI analysis (JSON) |
| confidence_score | Float | AI confidence (0-1) |
| crop_type | String(100) | Identified crop |
| severity_level | String(50) | Disease severity |
| recommendations | Text | Treatment advice |

#### `market_prices`
| Column | Type | Description |
|---|---|---|
| id | Integer (PK) | Unique ID |
| crop_name | String(100) | Crop commodity |
| market_name | String(100) | Market/mandi name |
| region | String(100) | Geographic region |
| price | Float | Price per unit |
| unit | String(20) | kg / quintal |
| date | Date | Price date |
| source | String(100) | Data source |

#### `crop_plans`
| Column | Type | Description |
|---|---|---|
| id | Integer (PK) | Unique ID |
| user_id | FK → users | Owner |
| crop_type | String(100) | Crop being grown |
| variety | String(100) | Crop variety |
| start_date | Date | Planting date |
| expected_harvest | Date | Harvest date |
| area | Float | Area in acres |
| planting_method | String(50) | Transplant / Direct |

#### `crop_tasks`
| Column | Type | Description |
|---|---|---|
| id | Integer (PK) | Unique ID |
| plan_id | FK → crop_plans | Parent plan |
| title | String(200) | Task name |
| due_date | Date | Task deadline |
| status | String(20) | pending / completed |
| category | String(50) | Task category |

#### `forum_threads` & `forum_posts`
- Forum threads with categories, tags, pins, and locks
- Posts with likes and solution marking

#### `chat_messages`
- Real-time room-based messages with reactions and reply threading

#### `documents`
- User-uploaded documents with file metadata

#### `weather_alerts`, `user_points`, `referrals`, `otp_verifications`
- Supporting tables for notifications, gamification, and security

---

## 8. Feature Enhancements (Future Scope)

### Short-Term Enhancements (0–6 months)
1. **SMS/WhatsApp Notifications**: Send critical alerts via SMS or WhatsApp using Twilio API
2. **Voice Assistant**: Voice-to-text input for farmers with low literacy levels
3. **Offline Mode (PWA)**: Progressive Web App support for use in low-connectivity areas
4. **Push Notifications**: Browser push notifications for weather and market alerts
5. **Multi-language AI Responses**: Full support for Tamil, Telugu, Kannada, Hindi, etc.

### Medium-Term Enhancements (6–12 months)
6. **Satellite Crop Monitoring**: Integration with satellite imagery APIs for field health mapping
7. **Government Scheme Integration**: Auto-fetch and display relevant government subsidies
8. **Mobile App (React Native)**: iOS and Android application
9. **IoT Sensor Integration**: Connect soil/temperature sensors for real-time monitoring
10. **Marketplace Module**: Platform for farmers to post crops for sale and buyers to purchase

### Long-Term Enhancements (12+ months)
11. **Predictive Analytics**: Machine learning models for yield prediction
12. **AI Crop Calendar**: Auto-generate optimal planting schedules based on climate + soil data
13. **Supply Chain Integration**: Direct connection to mandis (wholesale markets)
14. **Insurance Integration**: Crop insurance recommendation and claim assistant
15. **Drone Surveillance**: Integration with drone footage analysis for large farms

---

## 9. Conclusion

AgroBot is a comprehensive, AI-powered agricultural platform designed specifically for Indian farmers. By combining:

- **Google Gemini's advanced AI** for natural language understanding and computer vision
- **Real-time weather integration** for actionable farming recommendations
- **Community collaboration tools** to share knowledge among farmers
- **Structured crop planning** to optimize farm operations
- **Market intelligence** to empower fair pricing decisions

...AgroBot bridges the information gap between modern technology and the farming community.

The platform is built on a solid, scalable technology stack (Python Flask + SQLAlchemy + SocketIO) that can grow from a local SQLite deployment to a full-scale PostgreSQL production system.

**Impact**: AgroBot has the potential to improve farm productivity by 20–40%, reduce crop losses due to disease by 30%, and help farmers earn 15–25% more by making informed market decisions.

The project demonstrates how AI can be made practical, accessible, and beneficial for one of the most important sectors of the Indian economy – agriculture.

---

## 10. Sample Output

### Sample AI Chatbot Response
**User Query**: "My tomato plants have yellow leaves, what should I do?"

**AgroBot Response**:
```
🌿 Tomato Leaf Yellowing – Possible Causes & Solutions

1. **Nitrogen Deficiency** (Most Common)
   - Symptom: Older/lower leaves turn yellow first
   - Solution: Apply Urea (46-0-0) at 25 kg/acre

2. **Magnesium Deficiency**
   - Symptom: Yellowing between green veins
   - Solution: Foliar spray of Magnesium Sulphate (2%)

3. **Early Blight (Fungal)**
   - Symptom: Yellow halos around brown spots
   - Solution: Mancozeb 75 WP at 2.5 g/liter

⚠️ Precautions:
- Avoid overhead irrigation to prevent fungal spread
- Remove and destroy affected leaves

📞 For region-specific advice:
Contact Kisan Call Center: 1551
```

### Sample Image Analysis Output
```json
{
  "health_status": "Disease Detected – Early Blight",
  "confidence": 0.85,
  "analysis": "The image shows characteristic early blight symptoms with 
               concentric ring patterns and yellow chlorotic halos.",
  "recommendations": [
    "Apply Mancozeb fungicide immediately",
    "Ensure proper plant spacing for air circulation",
    "Remove infected leaves and destroy them",
    "Avoid wetting foliage during irrigation"
  ]
}
```

### Sample Weather Recommendation Output
```
🌤️ Weather: 28°C | Humidity: 72% | Rain Probability: 65%

Farming Recommendations:
✅ Irrigation  → Rain expected – skip irrigation today
⚠️ Pest Alert  → High humidity + rain = fungal risk. Apply preventive spray
📅 Fertilizer  → Wait for calmer weather before foliar spray
🚨 Harvest     → Rain expected – harvest ripe crops immediately
```

### Application Routes Summary

| Route | Method | Description |
|---|---|---|
| `/` | GET | Landing/Home page |
| `/register` | GET, POST | User registration |
| `/login` | GET, POST | User login |
| `/dashboard` | GET | User dashboard |
| `/chat` | GET | AI Chatbot interface |
| `/api/chat` | POST | AI chat API endpoint |
| `/api/analyze-image` | POST | Image analysis endpoint |
| `/weather` | GET | Weather forecast page |
| `/api/weather` | GET | Weather data API |
| `/market` | GET | Market prices page |
| `/api/market-prices` | GET | Market price API |
| `/pest-database` | GET | Pest encyclopedia |
| `/community` | GET | Community forum |
| `/chat-community` | GET | Real-time community chat |
| `/crop-planner` | GET | Crop planning tool |
| `/docs` | GET | Document management |
| `/notifications` | GET | User notifications |
| `/profile` | GET, POST | User profile |
| `/admin` | GET | Admin dashboard |

---

*Documentation generated for AgroBot v0.1.0 | March 2026*
