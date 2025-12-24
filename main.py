import os
import base64
import json
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from inference_sdk import InferenceHTTPClient

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# --- Upload configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB file size limit
app.secret_key = os.urandom(24)  # For flash messaging and session security

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Roboflow API configuration ---
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE_ID = os.getenv("ROBOFLOW_WORKSPACE_ID")
ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID")

# Initialize the Roboflow inference client
client = None
try:
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    )
    print("Roboflow client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Roboflow client: {e}")

# --- Utility function to check allowed file types ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Run Roboflow model inference workflow ---
def run_roboflow_workflow(image_path):
    try:
        if not client:
            print("Roboflow client not initialized.")
            return None

        # Read and encode the image in base64
        with open(image_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode('utf-8')

        # Send image to Roboflow workflow API
        result = client.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE_ID,
            workflow_id=ROBOFLOW_WORKFLOW_ID,
            images={"image": image_b64}
        )

        # Debug: print the raw response
        print("\n🔍 Roboflow Raw Response:\n" + json.dumps(result, indent=4), "\n")

        # Parse the class name from predictions if available
        if isinstance(result, list) and len(result) > 0:
            first_result = result[0]

            predictions = first_result.get("$steps.model.predictions", {}).get("predictions", [])
            if predictions and "class" in predictions[0]:
                return predictions[0]["class"]

            # Fallback if 'top_class' is available
            if "top_class" in first_result:
                return first_result["top_class"]

        return None

    except Exception as e:
        print(f"Roboflow workflow error: {e}")
        return None

# --- Route: Home page (upload form and result handling) ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            flash('No image file selected.', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            predicted_class = run_roboflow_workflow(filepath)
            if predicted_class:
                # Redirect to the page for the predicted biscuit class
                return redirect(url_for('show_recipe', biscuit=predicted_class))

            flash("Prediction failed or returned no result.", "error")
        else:
            flash('Invalid file type.', 'error')

    return render_template('index.html')

# --- Route: Serve uploaded files (if needed) ---
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Route: Show dedicated biscuit page based on prediction ---
@app.route('/biskut/<biscuit>')
def show_recipe(biscuit):
    try:
        return render_template(f'biscuits/{biscuit}.html')
    except:
        return f"<h1>No recipe page found for '{biscuit}'</h1>", 404

# --- Run the Flask development server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
