import cv2
import numpy as np

# Načtení předtrénovaného modelu YOLO
yolo_net = cv2.dnn.readNet("data_yolo/yolov3.weights", "data_yolo/yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Načtení COCO labels
with open("data_yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Funkce pro zpracování rámce z kamery
def process_frame(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Filtrujeme slabé predikce
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Spuštění kamery a detekce objektů
cap = cv2.VideoCapture(0)  # Spustíme kameru (0 = výchozí kamera)
while True:
    ret, frame = cap.read()  # Načteme aktuální snímek z kamery
    if not ret:
        break

    frame = process_frame(frame)

    # Zobrazení rámce
    cv2.imshow("Object Detection with YOLO", frame)
    print("1/1 ━━━━━━━━━━━━━━━━━━━━ 0s")

    # Umožní ukončit program stisknutím klávesy 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Uvolníme kameru
cv2.destroyAllWindows()  # Zavřeme všechna okna
