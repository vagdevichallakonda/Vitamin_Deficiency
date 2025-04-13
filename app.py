from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import os
import image_fuzzy_clustering as fem
import label_image
from video_detect import detect_best_face
from record_video import record

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'img')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

VITAMIN_INFO = {
    "Vitamin A": " → Deficiency of vitamin A is associated with significant morbidity and mortality from common childhood infections, and is the world's leading preventable cause of childhood blindness...",
    "Vitamin B": " → Vitamin B12 deficiency may lead to a reduction in healthy red blood cells (anaemia). The nervous system may also be affected...",
    "Vitamin C": " → A condition caused by a severe lack of vitamin C in the diet. Found in citrus fruits and vegetables. Scurvy results from a deficiency...",
    "Vitamin D": " → Vitamin D deficiency can lead to a loss of bone density, contributing to osteoporosis and fractures...",
    "Vitamin E": " → Needs fat for the digestive system to absorb it. Deficiency can cause nerve and muscle damage, weakness, and vision problems..."
}


# ------------------------- Helper Functions -------------------------
def save_img(img, filename):
    path = os.path.join('static/images', filename)
    Image.open(img).save(path)
    return path


def load_image(image_path):
    return label_image.main(image_path)


def process(image_path):
    try:
        print("[INFO] Performing image clustering...")
        fem.plot_cluster_img(image_path, 3)
        clustered_path = 'static/images/orig_image.jpg'
        result = load_image(clustered_path)
        return result.title() if result else "Prediction failed"
    except Exception as e:
        print(f"[ERROR] Image processing failed: {e}")
        return "Prediction failed"


def append_info(result):
    return result + VITAMIN_INFO.get(result, " → No additional details found.")


def handle_result(result, image_path):
    if not result or result == "Prediction failed":
        return jsonify({'error': 'Prediction failed'}), 500

    result = append_info(result)
    print("[RESULT]", result)

    if os.path.exists(image_path):
        os.remove(image_path)

    return result


# ------------------------- Routes -------------------------
@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/upload')
def upload():
    return render_template('index1.html')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({'error': 'No file uploaded'}), 400

        filename = secure_filename(file.filename)
        saved_path = save_img(file, filename)
        result = process(saved_path)
        return handle_result(result, saved_path)

    except Exception as e:
        print(f"[ERROR] /upload_image failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload_video', methods=['POST'])
def upload_video_route():
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({'error': 'No video uploaded'}), 400

        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        print(f"[INFO] Video saved at {video_path}")
        detected_path = detect_best_face(video_path)
        return process_detected_face(detected_path)

    except Exception as e:
        print(f"[ERROR] /upload_video failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/record_video', methods=['GET', 'POST'])
def record_video_route():
    try:
        video_path = record()
        print(f"[INFO] Video recorded at {video_path}")
        detected_path = detect_best_face(video_path)
        return process_detected_face(detected_path)

    except Exception as e:
        print(f"[ERROR] /record_video failed: {e}")
        return jsonify({'error': str(e)}), 500


# ------------------------- Utility Handler -------------------------
def process_detected_face(detected_path):
    if not detected_path or not os.path.exists(detected_path):
        return jsonify({'error': 'No face detected'}), 400

    result = process(detected_path)
    return handle_result(result, detected_path)


# ------------------------- Main -------------------------
if __name__ == '__main__':
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000')
    app.run(debug=True)
