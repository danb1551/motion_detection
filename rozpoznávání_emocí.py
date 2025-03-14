import cv2
from deepface import DeepFace

# Spuštění kamery
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detekce emocí pomocí DeepFace
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    
    # Získání nejvíce pravděpodobné emoce
    dominant_emotion = result[0]['dominant_emotion']
    confidence = result[0]['emotion'][dominant_emotion]
    
    # Zobrazení výsledků na obrazovce
    cv2.putText(frame, f"Emotion: {dominant_emotion} ({confidence:.2f})", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Zobrazení rámce
    cv2.imshow("Emotion Detection", frame)

    # Umožní ukončit program stisknutím klávesy 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Uvolníme kameru
cv2.destroyAllWindows()  # Zavřeme všechna okna
