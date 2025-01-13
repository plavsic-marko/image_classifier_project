import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Učitavamo unapred trenirani model MobileNetV2
model = MobileNetV2(weights="imagenet")


def predict_image(image_path):
    """
    Funkcija koja analizira sliku i vraća top 3 predikcije.
    """
    # Učitavanje slike i prilagođavanje veličine
    # Veličina koju model očekuje
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)  # Pretvaranje slike u numpy array
    img_array = preprocess_input(img_array)  # Normalizacija podataka
    img_array = tf.expand_dims(img_array, axis=0)  # Dodavanje batch dimenzije

    # Predikcija
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(
        predictions, top=3)  # Vraćamo top 3 rezultata
    return decoded_predictions
