from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
from PIL import Image
import io

# ---------------- FLASK SETUP ----------------
app = Flask(__name__)
app.secret_key = 'supersecretkey'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------------- DATABASE ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# ---------------- LOAD MODEL ----------------
model = load_model("code/Vgg16/blood_group_detection_vgg16.h5")

print("MODEL OUTPUT SHAPE:", model.output_shape)

# Same order as training folders
CLASSES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# ---------------- PERFECT PREPROCESS ----------------
def preprocess_image(file):

    # Read raw uploaded bytes (IMPORTANT)
    img_bytes = file.read()

    # Open without browser modifications
    img = Image.open(io.BytesIO(img_bytes))

    print("Original Mode:", img.mode)

    # Fingerprint dataset = grayscale → enforce
    img = img.convert('L')

    # Resize exactly like training
    img = img.resize((256, 256))

    # Convert to numpy
    img_array = np.array(img)

    # Convert grayscale → 3 channel (VGG16 needs 3)
    img_array = np.stack((img_array,) * 3, axis=-1)

    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Same preprocessing used in training
    img_array = preprocess_input(img_array)

    return img_array

# ---------------- ROUTES ----------------
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/about')
def about():
    return render_template('about.html')

# ---------------- SIGNUP ----------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash('Signup successful. Please login.')
        return redirect(url_for('login'))

    return render_template('signup.html')

# ---------------- LOGIN ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and user.password == password:
            session['user_id'] = user.id
            flash('Login successful')
            return redirect(url_for('prediction'))
        else:
            flash('Invalid credentials')

    return render_template('login.html')

# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully')
    return redirect(url_for('landing'))

# ---------------- PREDICTION ----------------
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():

    if 'user_id' not in session:
        flash('Please log in first')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    username = user.username

    if request.method == 'POST':

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'Empty file'})

        try:
            processed_img = preprocess_image(file)

            # Predict
            result = model.predict(processed_img)

            print("Raw Model Output:", result)

            predicted_class = int(np.argmax(result))
            confidence = float(result[0][predicted_class] * 100)

            if predicted_class < 0 or predicted_class >= len(CLASSES):
                return jsonify({'error': 'Invalid prediction index'})

            prediction = CLASSES[predicted_class]

            return jsonify({
                'prediction': prediction,
                'confidence': f"{confidence:.2f}%"
            })

        except Exception as e:
            print("ERROR:", str(e))
            return jsonify({'error': str(e)})

    return render_template('prediction.html', username=username)

# ---------------- MAIN ----------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    app.run(debug=True)

