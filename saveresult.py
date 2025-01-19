from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
from google.oauth2 import service_account
from google.cloud import firestore

app = FastAPI()

# Konfigurasi model TFLite
MODEL_PATH = "model.tflite"
LABELS = ["organik", "berbahaya", "non-organik"]

try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

# Detail input dan output dari model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Konfigurasi Firestore
FIREBASE_PROJECT = "jinam-446907"
FIREBASE_CREDENTIALS_PATH = "jinam-446907-firebase-adminsdk-h5ozk-83579d2459.json"

firebase_credentials = service_account.Credentials.from_service_account_file(FIREBASE_CREDENTIALS_PATH)
firestore_client = firestore.Client(project=FIREBASE_PROJECT, credentials=firebase_credentials)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Classification API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Membaca file yang diunggah
        content = await file.read()

        # Decode dan preprocessing gambar
        image = tf.io.decode_image(content, channels=3)
        image = tf.image.resize(image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))

        # Normalize dan cast ke UINT8 jika diperlukan
        if input_details[0]['dtype'] == np.uint8:
            image = image / 255.0  # Normalisasi ke rentang [0, 1]
            image = tf.cast(image * 255.0, tf.uint8)  # Ubah ke [0, 255] dan cast ke UINT8
        else:
            image = tf.cast(image, tf.float32)

        # Menambahkan dimensi batch
        image = tf.expand_dims(image, axis=0)

        # Lakukan inferensi
        interpreter.set_tensor(input_details[0]['index'], image.numpy())
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Konversi output model ke float32 jika tipe datanya uint8
        if output_details[0]['dtype'] == np.uint8:
            scale, zero_point = output_details[0]['quantization']
            output_data = scale * (output_data.astype(np.float32) - zero_point)

        # Softmax untuk memastikan probabilitas
        probabilities = tf.nn.softmax(output_data[0]).numpy()

        # Dapatkan label prediksi dan confidence
        predicted_index = np.argmax(probabilities)
        predicted_label = LABELS[predicted_index]
        confidence = probabilities[predicted_index] * 100  # Konversi ke persentase

        # Simpan label prediksi ke Firestore
        doc_ref = firestore_client.collection("predictions").document()
        doc_ref.set({"label": predicted_label})

        return {
            "label": predicted_label,
            "confidence": round(confidence, 2),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
