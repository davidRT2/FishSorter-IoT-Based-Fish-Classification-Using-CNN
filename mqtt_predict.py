import paho.mqtt.client as mqtt
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Inisialisasi model
model_path = 'my_model3.h5'
model = load_model(model_path)

# Membuat ulang LabelEncoder
le = LabelEncoder()
le.classes_ = np.load("label_encoder_classes.npy")

# Fungsi untuk melakukan prediksi
def predict_and_publish(client, userdata, message):
    try:
        # Mendapatkan gambar dalam format Base64 dari pesan MQTT
        base64_img_string = message.payload.decode("utf-8")

        # Mendekode gambar dari Base64
        _, base64_img_string = base64_img_string.split(',')
        img_bytes = base64.b64decode(base64_img_string)
        
        # Membuat objek BytesIO dari bytes
        img_array = image.img_to_array(img_bytes)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Melakukan prediksi menggunakan model
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_label = le.inverse_transform([predicted_class])[0]

        # Mengirim hasil prediksi ke topik "fd/pred"
        client.publish("fd/pred", f"Predicted class: {predicted_label}")
    except Exception as e:
        print(f"Error: {str(e)}")

# Konfigurasi MQTT
mqtt_broker = "test.mosquitto.org"
mqtt_topic = "fd/send"

client = mqtt.Client()
client.on_message = predict_and_publish

# Menghubungkan ke broker MQTT
client.connect(mqtt_broker, 1883, 60)
client.subscribe(mqtt_topic)

# Loop forever
client.loop_forever()
