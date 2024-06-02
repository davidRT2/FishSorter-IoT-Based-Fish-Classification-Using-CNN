import base64
from PIL import Image
from io import BytesIO

import os

def image_to_base64(img_path):
    # Membuka gambar
    img = Image.open(img_path)

    # Mengubah gambar ke mode RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Membuat objek BytesIO
    img_io = BytesIO()

    # Menyimpan gambar ke objek BytesIO
    img.save(img_io, format='JPEG')

    # Mengubah objek BytesIO menjadi bytes
    img_bytes = img_io.getvalue()

    # Mengubah bytes menjadi string base64
    base64_img_string = base64.b64encode(img_bytes).decode('utf-8')

    return base64_img_string

img_path = 'mujaireal.jpg'
base64_img_string = image_to_base64(img_path)

# Menyimpan string base64 ke file teks
filename = 'output'
extension = '.txt'
i = 0
while os.path.exists(f'{filename}{i}{extension}'):
    i += 1

with open(f'{filename}{i}{extension}', 'w') as f:
    f.write(base64_img_string)