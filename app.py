# Importing Libraries
import streamlit as st
import io
import numpy as np
from PIL import Image 
import tensorflow as tf
import efficientnet.tfkeras as efn

# Title and Description
st.title('Leaf Deseases Detection 🍁')
st.write("Cukup unggah gambar daun tanaman anda dan dapatkan prediksi Apakah Tanaman tersebut sehat atau tidak!")
st.write("")


gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Loading Model
model = tf.keras.models.load_model("model.h5")

# Upload the image
uploaded_file = st.file_uploader("Pilih file Gambar: (Dengan format png atau jpg)", type=["png", "jpg"])


predictions_map = {0:"sehat", 1:"memiliki beberapa penyakit", 2:"memiliki bercak", 3:"memiliki keropeng"}

if uploaded_file is not None:

    image = Image.open(io.BytesIO(uploaded_file.read()))

    st.image(image, use_column_width=True)

    # Image Preprocessing
    resized_image = np.array(image.resize((512, 512)))/255.

    # Adding batch dimension
    image_batch = resized_image[np.newaxis, :, :, :]

    # Getting the predictions fom the model
    predictions_arr = model.predict(image_batch)

    predictions = np.argmax(predictions_arr)

    result_text = f"Daun tanaman {predictions_map[predictions]} dengan kemungkinan sebesar {int(predictions_arr[0][predictions]*100)}%"

    if predictions == 0:
        st.success(result_text)
    else:
        st.error(result_text)