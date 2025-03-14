import cv2
import numpy as np
import time

# Načtení modelu YOLO
net = cv2.dnn.readNet("data_yolo/yolov3.weights", "data_yolo/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Načtení klasifikátorů
with open("data_yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Spuštění kamery
cap = cv2.VideoCapture(0)

# Načteme předchozí rámec pro detekci pohybu
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

# Sledovací algoritmus (CSRT)
tracker = None
object_bbox = None
motion_start_time = None

while True:
    time.sleep(0.1)
    ret, frame = cap.read()
    if not ret:
        break

    # Převedení na šedý obraz a rozmazání pro detekci pohybu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Rozdíl mezi předchozím a aktuálním rámcem
    delta_frame = cv2.absdiff(prev_gray, gray)
    threshold = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]

    # Detekce kontur pohybu
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pokud pohyb není detekován, pokračujeme
    if len(contours) == 0:
        prev_gray = gray
        continue

    # Detekce pohybu byla zjištěna, nyní spustíme YOLO pro rozpoznání objektu
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)

        # Pokud je objekt detekován, použijeme YOLO pro rozpoznání objektu
        roi = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(roi, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    w_box = int(detection[2] * w)
                    h_box = int(detection[3] * h)
                    x_box = center_x - w_box // 2
                    y_box = center_y - h_box // 2
                    boxes.append([x + x_box, y + y_box, w_box, h_box])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])

                # Pokud je detekován nový objekt, začneme sledovat
                if tracker is None:
                    tracker = cv2.TrackerMIL.create()  # Můžeš změnit na jiný tracker

                    object_bbox = (x, y, w, h)
                    tracker.init(frame, object_bbox)
                    motion_start_time = time.time()

        # Pokud je sledování objektu aktivní
        if tracker is not None:
            success, bbox = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Kontrola, zda objekt není v pohybu 5 sekund
                if time.time() - motion_start_time > 5:
                    cv2.putText(frame, "No movement detected for 5 seconds", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    tracker = None
                    motion_start_time = None

            else:
                cv2.putText(frame, "Tracking failure", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    prev_gray = gray
    cv2.imshow("Motion and Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
