# Full-Stack AI Plant Disease Detector

A web application that uses on-device AI for real-time plant disease detection and a secure backend to provide AI-generated treatment plans.

## 🌟 Key Features

**Fast, In-Browser AI:** Uses ONNX.js to run a trained PyTorch (MobileNetV2) model directly in the user's browser. No server-side GPU is needed for detection.

**High-Accuracy Diagnosis:** Built using Transfer Learning on the 38-class PlantVillage dataset to achieve high-confidence predictions.

**AI-Powered Treatment Plans:** Securely calls the Gemini API from a Flask backend to provide actionable, step-by-step treatment advice for the detected disease.

**Secure & Scalable:** All API keys are hidden in a .env file on the backend. The app is served as a single, monolithic Flask application, making it easy to deploy.

**Fully Responsive:** Built with Tailwind CSS (via CDN) for a clean, mobile-first user interface.

## 🛠️ Tech Stack

This project is a full-stack monolithic application.

### Frontend (In-Browser):

**React:** For the dynamic user interface (loaded via CDN).

**ONNX.js:** The runtime for executing the AI model in the browser.

**Tailwind CSS:** For all styling.

**Babel (in-browser):** To transpile the JSX for development.

### Backend (Server-Side):

**Flask:** A Python web server that serves both the React frontend and the secure backend API.

**Gunicorn:** A production-ready web server (for deployment).

### Artificial Intelligence:

**PyTorch:** Used to train and fine-tune the MobileNetV2 model (see train.py).

**ONNX:** The open format used to export the PyTorch model for the web.

**Gemini (2.5 Flash):** The Large Language Model used for generating treatment plans.
