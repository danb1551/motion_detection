import cv2
import numpy as np
import time

# Spuštění kamery
cap = cv2.VideoCapture(0)
cas = 2
# Načtení předchozího rámce pro detekci pohybu
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

# Sledovací algoritmus (TrackerMIL)
tracker = None
motion_start_time = None
object_bbox = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Převedení na šedý obraz a rozmazání
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Rozdíl mezi předchozím a aktuálním rámcem
    delta_frame = cv2.absdiff(prev_gray, gray)
    threshold = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Detekce kontur pohybu
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Pokud je detekován pohyb
    if len(contours) > 0 and tracker is None:  # Detekce pohybu pouze pokud není aktivní sledování
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Zelený obdélník pro pohyb
            
            # Pokud není sledování aktivní, začneme sledovat objekt
            if tracker is None:
                tracker = cv2.TrackerMIL.create()  # Vytvoření sledovače
                object_bbox = (x, y, w, h)  # Uložení souřadnic objektu
                tracker.init(frame, object_bbox)  # Inicializace sledovače
                motion_start_time = time.time()  # Čas, kdy byl detekován pohyb

    # Pokud je sledování objektu aktivní
    if tracker is not None:
        success, bbox = tracker.update(frame)  # Sledování objektu
        if success:
            (x, y, w, h) = [int(v) for v in bbox]  # Získání nových souřadnic objektu
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Červený obdélník pro sledování
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Pokud objekt není v pohybu 5 sekund, přestaneme sledovat
            if time.time() - motion_start_time > cas:
                cv2.putText(frame, f"No movement for {cas} seconds", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                tracker = None  # Zastavení sledování
                motion_start_time = None
        else:
            cv2.putText(frame, "Tracking failure", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Aktualizace předchozího rámce
    prev_gray = gray

    # Zobrazení výsledků
    cv2.imshow("Motion Detection and Object Tracking", frame)

    # Ukončení programu stisknutím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()