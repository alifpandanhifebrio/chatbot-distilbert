from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import firebase_admin
from firebase_admin import credentials, db
from chatbot import get_response

load_dotenv()

# Initialize Firebase Admin
cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred, {"databaseURL": "https://chatbot-v2-423214-default-rtdb.firebaseio.com/"})

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', False)

@app.route("/")
def home():
    return jsonify({"home": "Hello!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("message")

    if not text:
        return jsonify({"error": "No message provided"}), 400

    response = get_response(text)
    if not response:
        return jsonify({"error": "Jawaban tidak ditemukan"}), 500

    ref = db.reference('messages')
    ref.push({
        'message': text
    })

    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run()
