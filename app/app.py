# Importing essential libraries and modules

import os
import sqlite3
from markupsafe import Markup
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = getattr(config, "weather_api_key", "") or ""
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    city_name = (city_name or "").strip()
    if not city_name or not api_key:
        return None

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    try:
        response = requests.get(complete_url, timeout=10)
        x = response.json()
        if str(x.get("cod")) == "200" and isinstance(x.get("main"), dict):
            y = x["main"]
            temperature = round((y["temp"] - 273.15), 2)
            humidity = y["humidity"]
            return temperature, humidity
    except Exception:
        pass
    return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    with torch.no_grad():
        yb = model(img_u)
        probs = torch.softmax(yb, dim=1)
        conf, preds = torch.max(probs, dim=1)
    prediction = disease_classes[preds[0].item()]
    confidence = float(conf[0].item())
    return prediction, confidence

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'plantify-secret-key-change-in-production')

# ── Flask-Login ──────────────────────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'warning'

# ── Gate the app behind authentication ───────────────────────────────────────
@app.before_request
def require_login_for_app():
    """
    Force authentication before accessing any app page.
    Allows only auth endpoints + static assets without login.
    """
    if current_user.is_authenticated:
        return None

    # If Flask hasn't matched an endpoint yet, let it 404 naturally.
    if request.endpoint is None:
        return None

    allowed_endpoints = {
        'login',
        'register',
        'static',
    }
    if request.endpoint in allowed_endpoints:
        return None

    return redirect(url_for('login', next=request.full_path))

# ── User model ───────────────────────────────────────────────────────────────
class User(UserMixin):
    def __init__(self, id, name, email, password_hash):
        self.id = id
        self.name = name
        self.email = email
        self.password_hash = password_hash

# ── SQLite helpers ────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'users.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL
            )
        ''')
        conn.commit()

init_db()

def get_user_by_id(user_id):
    with get_db() as conn:
        row = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    return User(row['id'], row['name'], row['email'], row['password_hash']) if row else None

def get_user_by_email(email):
    with get_db() as conn:
        row = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    return User(row['id'], row['name'], row['email'], row['password_hash']) if row else None

def create_user(name, email, password):
    pw_hash = generate_password_hash(password)
    with get_db() as conn:
        conn.execute('INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)',
                     (name, email, pw_hash))
        conn.commit()

@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(int(user_id))

# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        name  = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        pwd   = request.form.get('password', '')
        cpwd  = request.form.get('confirm_password', '')

        if not name or not email or not pwd:
            flash('All fields are required.', 'danger')
        elif pwd != cpwd:
            flash('Passwords do not match.', 'danger')
        elif len(pwd) < 6:
            flash('Password must be at least 6 characters.', 'danger')
        elif get_user_by_email(email):
            flash('An account with that email already exists.', 'danger')
        else:
            create_user(name, email, pwd)
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html', title='Plantify - Register')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        pwd   = request.form.get('password', '')
        user  = get_user_by_email(email)
        if user and check_password_hash(user.password_hash, pwd):
            login_user(user, remember=request.form.get('remember') == 'on')
            next_page = request.args.get('next')
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(next_page or url_for('home'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html', title='Plantify - Login')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

# render home page


@ app.route('/')
@login_required
def home():
    title = 'Plantify - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
@login_required
def crop_recommend():
    title = 'Plantify - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
@login_required
def fertilizer_recommendation():
    title = 'Plantify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
@login_required
def crop_prediction():
    title = 'Plantify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        weather = weather_fetch(city)
        if weather is not None:
            temperature, humidity = weather
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template(
                'crop-result.html',
                prediction=final_prediction,
                title=title,
                N=N,
                P=P,
                K=K,
                ph=ph,
                rainfall=rainfall,
                city=city,
                temperature=temperature,
                humidity=humidity,
            )

        else:

            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
@login_required
def fert_recommend():
    title = 'Plantify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    basedir = os.path.abspath(os.path.dirname(__file__))
    fertilizer_csv = os.path.join(basedir, '..', 'Data-processed', 'fertilizer.csv')
    df = pd.read_csv(fertilizer_csv)

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template(
        'fertilizer-result.html',
        recommendation=response,
        title=title,
        crop_name=crop_name,
        N=N,
        P=P,
        K=K,
    )

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
@login_required
def disease_prediction():
    title = 'Plantify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return (request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            pred_label, confidence = predict_image(img)
            advice_html = Markup(str(disease_dic[pred_label]))

            # label format: "Apple___Cedar_apple_rust"
            plant = pred_label.split("___")[0].replace("_", " ")
            condition = pred_label.split("___")[1].replace("_", " ") if "___" in pred_label else pred_label

            return render_template(
                'disease-result.html',
                prediction=advice_html,
                title=title,
                plant=plant,
                condition=condition,
                confidence=round(confidence * 100, 1),
            )
        except:
            pass
    return render_template('disease.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5002"))
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
