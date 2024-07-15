from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import psycopg2
from chatbot import get_response

load_dotenv()

# Initialize Flask App
app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', False)

# Database configuration
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_PORT = os.getenv('DB_PORT', 5432)

# Create a connection to the database
conn = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    port=DB_PORT
)
cur = conn.cursor()

# Create table if it doesn't exist
cur.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id SERIAL PRIMARY KEY,
        message TEXT NOT NULL,
        response TEXT NOT NULL,
        timestamp TIMESTAMPTZ DEFAULT NOW()
    )
''')
conn.commit()

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

    # Insert message and response into the PostgreSQL database
    cur.execute('''
        INSERT INTO messages (message, response) VALUES (%s, %s)
    ''', (text, response))
    conn.commit()

    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run()

# Close the cursor and connection when the application is terminated
@app.teardown_appcontext
def close_db(error):
    if cur is not None:
        cur.close()
    if conn is not None:
        conn.close()
