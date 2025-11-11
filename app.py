import os
import requests  # For calling Gemini
from flask import Flask, request, jsonify, send_from_directory 
from dotenv import load_dotenv 


# --- NEW: Load environment variables ---
load_dotenv()

# Initialize the Flask app
app = Flask(__name__, static_folder='static')


# --- API Route (Stays the Same) ---
@app.route('/api/get_treatment', methods=['POST'])
def get_treatment_plan():
    """
    Handles the request for a treatment plan.
    This route calls the Gemini API from the *server*, which is secure.
    """
    data = request.json
    disease_name = data.get('disease')

    if not disease_name:
        return jsonify({'error': 'No disease name provided'}), 400

    # --- UPDATED: Read the key from the environment ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        # This handles the case where the key isn't found
        print("Error: GEMINI_API_KEY not found in .env file.")
        return jsonify({'error': 'Server configuration error: API key not found.'}), 500
    
    # --- FIX: Updated model to gemini-2.5-flash (from preview) ---
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

    # --- FIX: Refined prompt to ask for no markdown ---
    user_query = f"Provide a concise, step-by-step treatment plan for {disease_name}. Use a numbered list. Do not use markdown (like '###', '#', or '**'). Just provide the numbered steps."
    system_prompt = "You are a helpful assistant for a plant disease detection app. The user has a diagnosis and needs a concise, scannable, step-by-step treatment plan."

    payload = {
      "contents": [{"parts": [{"text": user_query}]}],
      # Use Google Search for factual, up-to-date treatment info
      "tools": [{"google_search": {}}],
      "systemInstruction": {
        "parts": [{"text": system_prompt}]
      },
    }

    try:
        # Make the API call from our server
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=60)
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
        
        result = response.json()
        
        # --- Extract Text ---
        candidate = result.get('candidates', [{}])[0]
        text_part = candidate.get('content', {}).get('parts', [{}])[0].get('text', '')

        if not text_part:
            # Handle cases where the API returns no text
            print("Gemini API Error: No text part found in response.")
            print("Full response:", result)
            raise Exception("Invalid API response structure or no content generated.")

        # --- Extract Sources ---
        sources = []
        grounding_metadata = candidate.get('groundingMetadata', {})
        if grounding_metadata and 'groundingAttributions' in grounding_metadata:
            for attribution in grounding_metadata['groundingAttributions']:
                if 'web' in attribution and 'uri' in attribution['web'] and 'title' in attribution['web']:
                    sources.append({
                        'uri': attribution['web']['uri'],
                        'title': attribution['web']['title']
                    })

        # Return both the plan and the sources
        return jsonify({
            'plan': text_part,
            'sources': sources
        })

    except requests.exceptions.RequestException as e:
        print(f"Gemini API request error: {e}")
        return jsonify({'error': f'Failed to contact Gemini API: {e}'}), 500
    except Exception as e:
        print(f"Gemini API processing error: {e}")
        return jsonify({'error': f'Failed to get treatment plan: {e}'}), 500

# --- NEW: React App Serving ---
@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_react_app(path):
    """
    Serves the static files for the React app.
    This includes index.html, .jsx, .onnx, and .onnx.data files.
    """
    try:
        return send_from_directory(app.static_folder, path)
    except FileNotFoundError:
        return send_from_directory(app.static_folder, 'index.html')

# --- Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)