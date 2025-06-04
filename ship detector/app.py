import os
import subprocess
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'sar_output/ship_predictions'
MODEL_PATH = os.path.abspath('sar_output/ship_detector_run/weights/best.pt')
YOLOV5_DIR = 'yolov5'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check and download YOLOv5 if not present
def setup_yolov5():
    if not os.path.exists(YOLOV5_DIR):
        try:
            # Clone YOLOv5 repository
            subprocess.run([
                'git', 'clone', 'https://github.com/ultralytics/yolov5.git', YOLOV5_DIR
            ], check=True)
            # Install YOLOv5 dependencies
            subprocess.run([
                'pip', 'install', '-r', os.path.join(YOLOV5_DIR, 'requirements.txt')
            ], check=True)
        except subprocess.CalledProcessError as e:
            return False, f"Failed to setup YOLOv5: {e.stderr}"
    return True, "YOLOv5 setup complete"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check and setup YOLOv5
    success, message = setup_yolov5()
    if not success:
        return render_template('index.html', error=message)

    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            # Verify file exists
            if not os.path.exists(filepath):
                return render_template('index.html', error=f"Uploaded file not found at {filepath}")

            # Change to YOLOv5 directory
            original_dir = os.getcwd()
            os.chdir(YOLOV5_DIR)

            # Run YOLOv5 detection with absolute path
            try:
                result = subprocess.run([
                    'python', 'detect.py',
                    '--weights', os.path.abspath(os.path.join(original_dir, MODEL_PATH)),
                    '--img', '640',
                    '--conf', '0.25',
                    '--source', os.path.abspath(os.path.join(original_dir, filepath)),
                    '--project', os.path.abspath(os.path.join(original_dir, 'sar_output')),
                    '--name', 'ship_predictions',
                    '--exist-ok'
                ], capture_output=True, text=True, check=True)

                # Extract prediction results (from detect.py output)
                prediction_output = result.stdout

                # Find the output image
                output_image_path = os.path.join(original_dir, 'sar_output/ship_predictions', unique_filename)
                if os.path.exists(output_image_path):
                    output_image = unique_filename
                else:
                    output_image = None

                # Return to original directory
                os.chdir(original_dir)

                return render_template('index.html', 
                                     prediction=prediction_output, 
                                     uploaded_image=unique_filename, 
                                     output_image=output_image)
            except subprocess.CalledProcessError as e:
                os.chdir(original_dir)
                return render_template('index.html', error=f"Error during inference: {e.stderr}")
        else:
            return render_template('index.html', error='Invalid file type')
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/sar_output/ship_predictions/<filename>')
def predicted_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)