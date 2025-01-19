from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

app = FastAPI()

# Load TFLite model
MODEL_PATH = "model.tflite"
LABELS = ["organik", "berbahaya", "non-organik"]

try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Classification API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        content = await file.read()

        # Decode and preprocess image
        image = tf.io.decode_image(content, channels=3)
        image = tf.image.resize(image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))

        # Normalize and cast to UINT8 if needed
        if input_details[0]['dtype'] == np.uint8:
            image = image / 255.0  # Normalize to range [0, 1]
            image = tf.cast(image * 255.0, tf.uint8)  # Scale back to [0, 255] and cast to UINT8
        else:
            image = tf.cast(image, tf.float32)  # Ensure FLOAT32 if required

        # Add batch dimension
        image = tf.expand_dims(image, axis=0)

        # Perform inference
        interpreter.set_tensor(input_details[0]['index'], image.numpy())
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get the predicted label
        predicted_index = np.argmax(output_data)
        predicted_label = LABELS[predicted_index]
        confidence = float(output_data[0][predicted_index])

        return {
            "label": predicted_label,
            "confidence": confidence,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
