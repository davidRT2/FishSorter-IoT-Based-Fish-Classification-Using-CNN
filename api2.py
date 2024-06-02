from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import os
import shutil
import paho.mqtt.client as mqtt

app = Flask(__name__)
CORS(app)  # Tambahkan baris ini untuk mengizinkan CORS

# Memuat model
model_path = 'my_model3.h5'
model = load_model(model_path)

# Membuat ulang LabelEncoder
le = LabelEncoder()
le.classes_ = np.load("label_encoder_classes.npy")

# Konfigurasi MQTT
mqtt_broker = "test.mosquitto.org"  # Ganti dengan broker MQTT yang sesuai
mqtt_port = 1883
mqtt_topic = "fd/pred"

# Membuat klien MQTT
mqtt_client = mqtt.Client()

# Menghubungkan klien ke broker
mqtt_client.connect(mqtt_broker, mqtt_port, 60)

# Memulai loop klien MQTT
mqtt_client.loop_start()

def publish_to_mqtt(label):
    # Mengirim label prediksi ke topik MQTT
    mqtt_client.publish(mqtt_topic, label)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mendapatkan file gambar dari form
        img_file = request.files['image']

        if not img_file:
            return jsonify({'error': 'No image file provided'})

        # Convert the FileStorage object to bytes
        img_bytes = img_file.read()

        # Create a BytesIO object from the bytes
        img_io = BytesIO(img_bytes)

        # Membaca gambar
        img = image.load_img(img_io, target_size=(150, 150))

        # Mengubah gambar menjadi array numpy
        img_array = image.img_to_array(img)

        # Menambahkan dimensi tambahan karena model menerima input dengan bentuk (batch_size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)

        # Normalisasi nilai piksel
        img_array /= 255.0

        # Melakukan prediksi
        prediction = model.predict(img_array)

        # Mendapatkan indeks kelas dengan probabilitas tertinggi
        predicted_class = np.argmax(prediction)

        # Mendapatkan label kelas dari indeks
        predicted_label = le.inverse_transform([predicted_class])[0]

        # Mencetak hasil prediksi
        print(f'Predicted class: {predicted_label}')

        publish_to_mqtt(predicted_label)

        save_dir = os.path.join('actuator_set', predicted_label)
        os.makedirs(save_dir, exist_ok=True)

        # Constructing the image path
        img_path = os.path.join(save_dir, 'image.jpg')

        # Check if the file already exists, if yes, increment the file name
        count = 1
        while os.path.exists(img_path):
            img_path = os.path.join(save_dir, f'image_{count}.jpg')
            count += 1

        # Save the image
        with open(img_path, 'wb') as f:
            f.write(img_bytes)

        return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    try:
        # Mendapatkan string base64 dari form
        base64_img_string = request.form.get('image')

        if not base64_img_string:
            return jsonify({'error': 'No image data provided'})

        _, base64_img_string = base64_img_string.split(',')

        # Mengubah string base64 menjadi bytes
        img_bytes = base64.b64decode(base64_img_string)

        # Membuat objek BytesIO dari bytes
        img_io = BytesIO(img_bytes)

        # Membaca gambar
        img = Image.open(img_io)

        # Mengubah ukuran gambar menjadi 150x150
        img = img.resize((150, 150))

        # Mengubah gambar menjadi array numpy
        img_array = image.img_to_array(img)

        # Menambahkan dimensi tambahan karena model menerima input dengan bentuk (batch_size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)

        # Normalisasi nilai piksel
        img_array /= 255.0

        # Melakukan prediksi
        prediction = model.predict(img_array)

        # Mendapatkan indeks kelas dengan probabilitas tertinggi
        predicted_class = np.argmax(prediction)

        # Mendapatkan label kelas dari indeks
        predicted_label = le.inverse_transform([predicted_class])[0]

        # Mencetak hasil prediksi
        print(f'Predicted class: {predicted_label}')

        publish_to_mqtt(predicted_label)

        save_dir = os.path.join('actuator_set', predicted_label)
        os.makedirs(save_dir, exist_ok=True)

        # Constructing the image path
        img_path = os.path.join(save_dir, 'image.jpg')

        # Check if the file already exists, if yes, increment the file name
        count = 1
        while os.path.exists(img_path):
            img_path = os.path.join(save_dir, f'image_{count}.jpg')
            count += 1

        # Save the image
        with open(img_path, 'wb') as f:
            f.write(img_bytes)

        return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='172.16.165.229', port=5000)