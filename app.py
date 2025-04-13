from flask import *
from werkzeug.utils import secure_filename

import os
import cv2

import image_fuzzy_clustering as fem
import record_video
import label_image

from PIL import Image

import secrets
from flask import url_for, current_app
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np




def load_image(image):
    text = label_image.main(image)
    return text

def save_img(img, filename):
    picture_path = os.path.join('static/images', filename)
    i = Image.open(img)
    i.save(picture_path)
    return picture_path

def process(image_path):
    print("[INFO] Performing image clustering...")
    fem.plot_cluster_img(image_path, 3)
    print("[INFO] Clustering completed.")

    clustered_path = 'static/images/orig_image.jpg'  # Assuming this is where clustering saves
    result = load_image(clustered_path)

    return result if result else "Prediction failed"






def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

app = Flask(__name__)
model = None

UPLOAD_FOLDER = os.path.join(app.root_path ,'static','img')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




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
    result=None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        original_filename = secure_filename(f.filename)
        original_path = save_img(f, original_filename)

        result = process(original_path)


        if not result or result == "Prediction failed":
            return jsonify({'error': 'Prediction failed'}), 500
        
        result = result.title()
        d = {
            "Vitamin A": " → Deficiency of vitamin A is associated with significant morbidity and mortality from common childhood infections, and is the world's leading preventable cause of childhood blindness. Vitamin A deficiency also contributes to maternal mortality and other poor outcomes of pregnancy and lactation.",
            'Vitamin B': " → Vitamin B12 deficiency may lead to a reduction in healthy red blood cells (anaemia). The nervous system may also be affected. Diet or certain medical conditions may be the cause. Symptoms are rare but can include fatigue, breathlessness, numbness, poor balance and memory trouble. Treatment includes dietary changes, B12 shots or supplements.",
            'Vitamin C': " → A condition caused by a severe lack of vitamin C in the diet. Vitamin C is found in citrus fruits and vegetables. Scurvy results from a deficiency of vitamin C in the diet. Symptoms may not occur for a few months after a person's dietary intake of vitamin C drops too low. Bruising, bleeding gums, weakness, fatigue and rash are among scurvy symptoms. Treatment involves taking vitamin C supplements and eating citrus fruits, potatoes, broccoli and strawberries.",
            'Vitamin D': " → Vitamin D deficiency can lead to a loss of bone density, which can contribute to osteoporosis and fractures (broken bones). Severe vitamin D deficiency can also lead to other diseases. In children, it can cause rickets. Rickets is a rare disease that causes the bones to become soft and bend.",
            "Vitamin E": " → Vitamin E needs some fat for the digestive system to absorb it. Vitamin E deficiency can cause nerve and muscle damage that results in loss of feeling in the arms and legs, loss of body movement control, muscle weakness, and vision problems. Another sign of deficiency is a weakened immune system."
        }
        result = result + d.get(result, " → No additional details found.")
        print(result)
        if os.path.exists(original_path): 
            os.remove(original_path)
        else:
            print(f"Warning: {original_path} not found, skipping deletion.")
        return result
        # return jsonify(f"Prediction: {result}")

    except Exception as e:
        print(f"[ERROR] /upload_image failed: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({'error': 'No file uploaded'}), 400

        video = request.files['file']
        filename = secure_filename(video.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        video.save(video_path)
        print(f"[INFO] Video saved at {video_path}")

        # Extract best face
        from video_detect import detect_best_face
        print("[INFO] Starting face detection...")
        best_face_path = detect_best_face(video_path)

        if not best_face_path or not os.path.exists(best_face_path):
            return jsonify({'error': 'No face detected'}), 400

        print(f"[INFO] Face detected and saved at {best_face_path}")
        detected_path ="static/Detected/detected.jpg"
        if os.path.exists(detected_path):
            result = process(detected_path)

            if not result or result == "Prediction failed":
                return jsonify({'error': 'Prediction failed'}), 500
        
            result = result.title()
            d = {
                "Vitamin A": " → Deficiency of vitamin A is associated with significant morbidity and mortality from common childhood infections, and is the world's leading preventable cause of childhood blindness. Vitamin A deficiency also contributes to maternal mortality and other poor outcomes of pregnancy and lactation.",
                'Vitamin B': " → Vitamin B12 deficiency may lead to a reduction in healthy red blood cells (anaemia). The nervous system may also be affected. Diet or certain medical conditions may be the cause. Symptoms are rare but can include fatigue, breathlessness, numbness, poor balance and memory trouble. Treatment includes dietary changes, B12 shots or supplements.",
                'Vitamin C': " → A condition caused by a severe lack of vitamin C in the diet. Vitamin C is found in citrus fruits and vegetables. Scurvy results from a deficiency of vitamin C in the diet. Symptoms may not occur for a few months after a person's dietary intake of vitamin C drops too low. Bruising, bleeding gums, weakness, fatigue and rash are among scurvy symptoms. Treatment involves taking vitamin C supplements and eating citrus fruits, potatoes, broccoli and strawberries.",
                'Vitamin D': " → Vitamin D deficiency can lead to a loss of bone density, which can contribute to osteoporosis and fractures (broken bones). Severe vitamin D deficiency can also lead to other diseases. In children, it can cause rickets. Rickets is a rare disease that causes the bones to become soft and bend.",
                "Vitamin E": " → Vitamin E needs some fat for the digestive system to absorb it. Vitamin E deficiency can cause nerve and muscle damage that results in loss of feeling in the arms and legs, loss of body movement control, muscle weakness, and vision problems. Another sign of deficiency is a weakened immune system."
            }
            result = result + d.get(result, " → No additional details found.")
            print(result)
            if os.path.exists(detected_path): 
                os.remove(detected_path)
            else:
                print(f"Warning: {detected_path} not found, skipping deletion.")
            return result
            #return f"Prediction: {result}"
        else:
            return "Error: Detected image not found."
    except Exception as e:
        print(f"[ERROR] /upload_image failed: {e}")
        return jsonify({'error': str(e)}), 500
        


@app.route('/record_video', methods=['GET','POST'])
def record_video_route():
    try:
        from record_video import record
        from video_detect import detect_best_face

        # Step 1: Record video from webcam
        print("[INFO] Starting video recording...")
        video_path = record()

        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': 'Video recording failed'}), 500

        print(f"[INFO] Video recorded and saved at {video_path}")

        # Step 2: Extract best face from video
        print("[INFO] Starting face detection...")
        best_face_path = detect_best_face(video_path)

        if not best_face_path or not os.path.exists(best_face_path):
            return jsonify({'error': 'No face detected from video'}), 400

        print(f"[INFO] Face detected and saved at {best_face_path}")
        
        detected_path ="static/Detected/detected.jpg"
        if os.path.exists(detected_path):
            result = process(detected_path)

            if not result or result == "Prediction failed":
                return jsonify({'error': 'Prediction failed'}), 500
        
            result = result.title()
            d = {
                "Vitamin A": " → Deficiency of vitamin A is associated with significant morbidity and mortality from common childhood infections, and is the world's leading preventable cause of childhood blindness. Vitamin A deficiency also contributes to maternal mortality and other poor outcomes of pregnancy and lactation.",
                'Vitamin B': " → Vitamin B12 deficiency may lead to a reduction in healthy red blood cells (anaemia). The nervous system may also be affected. Diet or certain medical conditions may be the cause. Symptoms are rare but can include fatigue, breathlessness, numbness, poor balance and memory trouble. Treatment includes dietary changes, B12 shots or supplements.",
                'Vitamin C': " → A condition caused by a severe lack of vitamin C in the diet. Vitamin C is found in citrus fruits and vegetables. Scurvy results from a deficiency of vitamin C in the diet. Symptoms may not occur for a few months after a person's dietary intake of vitamin C drops too low. Bruising, bleeding gums, weakness, fatigue and rash are among scurvy symptoms. Treatment involves taking vitamin C supplements and eating citrus fruits, potatoes, broccoli and strawberries.",
                'Vitamin D': " → Vitamin D deficiency can lead to a loss of bone density, which can contribute to osteoporosis and fractures (broken bones). Severe vitamin D deficiency can also lead to other diseases. In children, it can cause rickets. Rickets is a rare disease that causes the bones to become soft and bend.",
                "Vitamin E": " → Vitamin E needs some fat for the digestive system to absorb it. Vitamin E deficiency can cause nerve and muscle damage that results in loss of feeling in the arms and legs, loss of body movement control, muscle weakness, and vision problems. Another sign of deficiency is a weakened immune system."
            }
            result = result + d.get(result, " → No additional details found.")
            print(result)
            if os.path.exists(detected_path): 
                os.remove(detected_path)
            else:
                print(f"Warning: {detected_path} not found, skipping deletion.")
            return result
            #return f"Prediction: {result}"
        else:
            return "Error: Detected image not found."
        
    except Exception as e:
        print(f"[ERROR] /upload_image failed: {e}")
        return jsonify({'error': str(e)}), 500

# @app.route('/success', methods=['POST'])
# def success():
#     if request.method == 'POST':
#         i = request.form.get('cluster')
#         f = request.files['file']
#         original_pic_path = save_img(f, f.filename)
#         fem.plot_cluster_img(original_pic_path, i)
#     return render_template('success.html')

# @app.route('/index')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def upload1():
#     if request.method == 'POST':
#         f = request.files['file']
#         file_path = secure_filename(f.filename)
#         f.save(file_path)
#         result = load_image(file_path)
#         result = result.title()
#         d = {
#             "Vitamin A": " → Deficiency of vitamin A is associated with significant morbidity and mortality from common childhood infections, and is the world's leading preventable cause of childhood blindness. Vitamin A deficiency also contributes to maternal mortality and other poor outcomes of pregnancy and lactation.",
#             'Vitamin B': " → Vitamin B12 deficiency may lead to a reduction in healthy red blood cells (anaemia). The nervous system may also be affected. Diet or certain medical conditions may be the cause. Symptoms are rare but can include fatigue, breathlessness, numbness, poor balance and memory trouble. Treatment includes dietary changes, B12 shots or supplements.",
#             'Vitamin C': " → A condition caused by a severe lack of vitamin C in the diet. Vitamin C is found in citrus fruits and vegetables. Scurvy results from a deficiency of vitamin C in the diet. Symptoms may not occur for a few months after a person's dietary intake of vitamin C drops too low. Bruising, bleeding gums, weakness, fatigue and rash are among scurvy symptoms. Treatment involves taking vitamin C supplements and eating citrus fruits, potatoes, broccoli and strawberries.",
#             'Vitamin D': " → Vitamin D deficiency can lead to a loss of bone density, which can contribute to osteoporosis and fractures (broken bones). Severe vitamin D deficiency can also lead to other diseases. In children, it can cause rickets. Rickets is a rare disease that causes the bones to become soft and bend.",
#             "Vitamin E": " → Vitamin E needs some fat for the digestive system to absorb it. Vitamin E deficiency can cause nerve and muscle damage that results in loss of feeling in the arms and legs, loss of body movement control, muscle weakness, and vision problems. Another sign of deficiency is a weakened immune system."
#         }
#         result = result + d.get(result, " → No additional details found.")
#         print(result)
#         if os.path.exists(file_path): 
#             os.remove(file_path)
#         else:
#             print(f"Warning: {file_path} not found, skipping deletion.")
#         return result
#     return None

if __name__ == '__main__':
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000')
    app.run()