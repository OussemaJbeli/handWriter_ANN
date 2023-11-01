import os
import base64
from flask import Flask, request, jsonify, render_template
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('training.model')

UPLOAD_FOLDER = 'uploads/test/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

#********************** analyse and tretement of img*****************

def predict_digit(image_path):
    try:
        img = cv2.imread(image_path)[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        predicted_digit = int(np.argmax(prediction))
        return predicted_digit
    except Exception as e:
        print(f"Error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')
#********************** for fraowing with canvas*****************
@app.route('/predictDrow', methods=['POST'])
def predictDrow():
    try:
        data = request.get_json()
        if 'image' in data:
            image_data = data['image'].split(',')[1] 
            image_bytes = base64.b64decode(image_data)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
            with open(image_path, 'wb') as img_file:
                img_file.write(image_bytes)
        
        predicted_digit = predict_digit('uploads/test/image.png')
        if predicted_digit is not None:
            return jsonify({'predicted_digit': predicted_digit})
        else:
            return jsonify({'error': 'Error occurred during prediction.'})
    except Exception as e:
        return jsonify({'error': str(e)})
#********************** for img with path *****************
@app.route('/predictIMG', methods=['POST'])
def predictIMG():
    try:
        image = request.files['image']
        image_path = 'uploads/test/image.png'
        image.save(image_path)
        predicted_digit = predict_digit(image_path)
        if predicted_digit is not None:
            return jsonify({'predicted_digit': predicted_digit})
        else:
            return jsonify({'error': 'Error occurred during prediction.'})
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)