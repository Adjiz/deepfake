import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import os

# Load model gambar 
model = tf.keras.models.load_model('models/efficientnetv2b0_model.h5')

def predict_image(img_path, target_size=(200, 200)):
    img = Image.open(img_path).convert('RGB').resize(target_size)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]

    # anggap output sigmoid = probabilitas fake
    confidence_real = float(prediction)
    confidence_fake = float(1.0 - prediction)

    # tentukan label
    if confidence_fake >= 0.5:
        label = 'fake'
    else:
        label = 'real'

    return label, confidence_real, confidence_fake

def predict_video(file_path, target_size=(200, 200)):
    """Prediksi video dengan sampling frame"""
    video_capture = cv2.VideoCapture(file_path)

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // 20)  # Ambil 20 frame biar efisien

    predictions = []

    for i in range(0, total_frames, frame_step):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video_capture.read()
        if not ret:
            break

        # Preprocessing frame
    # Preprocessing frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame).resize(target_size)   
        img_array = image.img_to_array(frame)               
        img_array = np.expand_dims(img_array, axis=0)       
        img_array = img_array / 255.0                       


        # Prediksi frame → probabilitas fake
        prediction = model.predict(img_array, verbose=0)[0][0]
        predictions.append(prediction)

    video_capture.release()

    if not predictions:  # Kalau video kosong / error
        return "unknown", 0.0, 0.0

    # Ambil rata-rata prediksi
    avg_real_conf = float(np.mean(predictions))
    avg_fake_conf = float(1.0 - avg_real_conf)

    # Tentukan label
    label = "fake" if avg_fake_conf >= 0.5 else "real"

    return label, avg_real_conf, avg_fake_conf

