import cv2
import tensorflow as tf
import numpy as np

# Načtení předtrénovaného modelu MobileNetV2
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Funkce pro zpracování obrázků pro model
def prepare_image(image):
    image_resized = cv2.resize(image, (224, 224))  # Změna velikosti na 224x224 px
    image_array = tf.keras.preprocessing.image.img_to_array(image_resized)  # Převeď na array
    image_array = np.expand_dims(image_array, axis=0)  # Přidáme batch dimension
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)  # Předzpracování pro MobileNet
    return image_array

# Načtení objektů pro detekci
labels_path = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
labels = tf.keras.utils.get_file('imagenet_class_index.json', labels_path)
import json
with open(labels, 'r') as f:
    class_labels = json.load(f)

# Spuštění kamery a rozpoznávání objektů
cap = cv2.VideoCapture(0)  # Spustíme kameru (0 = výchozí kamera)
while True:
    ret, frame = cap.read()  # Načteme aktuální snímek z kamery
    if not ret:
        break

    # Připravíme snímek pro model
    prepared_image = prepare_image(frame)

    # Predikce pomocí MobileNetV2
    predictions = model.predict(prepared_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

    # Zobrazení výsledků
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        cv2.putText(frame, f'{label}: {score:.2f}', (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Zobrazení rámce
    cv2.imshow("Object Recognition", frame)

    # Umožní ukončit program stisknutím klávesy 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Uvolníme kameru
cv2.destroyAllWindows()  # Zavřeme všechna okna