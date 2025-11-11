Full-Stack AI Plant Disease Detector

A web application that uses on-device AI for real-time plant disease detection and a secure backend to provide AI-generated treatment plans.

(Note: You'll need to upload your screenshot—like the one you showed me—to a service like Imgur or a docs folder in your repo, and then replace the URL above.)

🌟 Key Features

Fast, In-Browser AI: Uses ONNX.js to run a trained PyTorch (MobileNetV2) model directly in the user's browser. No server-side GPU is needed for detection.

High-Accuracy Diagnosis: Built using Transfer Learning on the 38-class PlantVillage dataset to achieve high-confidence predictions.

AI-Powered Treatment Plans: Securely calls the Gemini API from a Flask backend to provide actionable, step-by-step treatment advice for the detected disease.

Secure & Scalable: All API keys are hidden in a .env file on the backend. The app is served as a single, monolithic Flask application, making it easy to deploy.

Fully Responsive: Built with Tailwind CSS (via CDN) for a clean, mobile-first user interface.

🛠️ Tech Stack

This project is a full-stack monolithic application.

Frontend (In-Browser):

React: For the dynamic user interface (loaded via CDN).

ONNX.js: The runtime for executing the AI model in the browser.

Tailwind CSS: For all styling.

Babel (in-browser): To transpile the JSX for development.

Backend (Server-Side):

Flask: A Python web server that serves both the React frontend and the secure backend API.

Gunicorn: A production-ready web server (for deployment).

Artificial Intelligence:

PyTorch: Used to train and fine-tune the MobileNetV2 model (see train.py).

ONNX: The open format used to export the PyTorch model for the web.

Gemini (2.5 Flash): The Large Language Model used for generating treatment plans.

🚀 How to Run Locally

Follow these steps to run the project on your local machine.

1. Prerequisites

Python 3.8+ and pip

A virtual environment tool (venv)

A Gemini API Key (from Google AI Studio)

2. Setup

Clone the repository:

git clone [https://github.com/your-username/your-project-name.git](https://github.com/your-username/your-project-name.git)
cd your-project-name


Set up the Python environment:

# Create a virtual environment
python -m venv venv

# Activate it
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


Add Your AI Models:
This repo does not include the large model files. You must add your trained models to the static/ folder:

static/model.onnx

static/model.onnx.data

Create Your Environment File:
Create a file named .env in the root of the project folder:

GEMINI_API_KEY="aIzaSy...your...key...here"


3. Run the Application

With your venv active, run the Flask server:

python app.py


The server will start, typically on http://127.0.0.1:5000.

Open your browser and navigate to http://127.0.0.1:5000 to use the app.

📁 Project Structure

This project is run as a "monolithic" Flask app, where the backend serves the frontend from a static folder.

/plant-disease-project/
├── app.py           <-- The Flask Server (Backend API + Frontend Server)
├── requirements.txt <-- Python dependencies
├── .env             <-- Your secret API key
│
├── static/          <-- ALL frontend files
│   ├── index.html   <-- Loads React/ONNX
│   ├── plant_disease_detector.jsx <-- The React app
│   ├── model.onnx   <-- Model graph
│   └── model.onnx.data <-- Model weights
│
└── (your training scripts...)


🔄 How It Works: Data Flow

User Visits: A user opens http://localhost:5000.

Flask Serves Frontend: The Flask server's serve_react_app route sends the static/index.html file and all its dependencies (React, ONNX.js, plant_disease_detector.jsx).

App Loads: The React app loads in the browser and automatically fetches the model.onnx files from the static folder to initialize the ONNX.js session.

Detection (In-Browser):

User uploads an image.

The React app pre-processes the image into a tensor (resize, normalize).

ONNX.js runs the model entirely in the browser and produces a diagnosis.

Treatment Plan (Backend Call):

User clicks "Get Treatment Plan."

The React app makes a fetch request to its own server (a relative path: /api/get_treatment).

Flask's @app.route('/api/get_treatment') catches this, securely reads the GEMINI_API_KEY from .env, and calls the Google Gemini API.

Flask returns the AI-generated text as JSON to the React app, which displays it.