import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configurar el estilo de la aplicación
st.markdown(
    """
    <style>
        body {
            background-color: #000000;
            color: #FFFFFF;
        }
        .main {
            background-color: #000000;
        }
        .st-bd {
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Título y descripción
st.title('Image Classifier: Sad or Happy Person.')
st.write("This simple application classifies whether a person in an uploaded image looks happy or sad.")

# Cargar el modelo
model_path = 'models/happysadmodel.h5'
model = load_model(model_path)

# Widget para cargar una imagen
upload_file = st.file_uploader("Upload image file (jpg)", type=["jpg"])

# Función para predecir y mostrar la imagen
def predict(image):
    # Verificar si la imagen se cargó correctamente
    if image is not None:
        # Mostrar la imagen original
        st.image(image, caption="Original Image", use_column_width=True)

        # Cambiar el tamaño de la imagen
        resize = tf.image.resize(image, (256, 256))

        # Normalizar y expandir dimensiones para la predicción
        input_image = np.expand_dims(resize / 255, 0)

        # Hacer la predicción
        yhat = model.predict(input_image)

        # Imprimir el resultado
        if yhat[0] > 0.5:
            result = "HAPPY"
        else:
            result = "SAD"

        # Mostrar el resultado de la predicción con un título
        st.write("## Prediction Result")
        st.write(f"Prediction: {result}")
    else:
        st.error("Please upload a valid image.")

# Botón para hacer la predicción
if upload_file is not None:
    # Leer la imagen usando OpenCV
    content = upload_file.getvalue()
    image = cv2.imdecode(np.frombuffer(content, np.uint8), 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predict(image_rgb)
