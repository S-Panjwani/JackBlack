from ultralytics import YOLO
import cv2
import os

modelpath = os.path.join(os.path.dirname(__file__), "Model", "yolov8m_synthetic.pt")
model = YOLO(modelpath)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("camera no working")
    exit()

camera.set(cv2.CAP_PROP_FPS, 15)

while True: 
    ret, frame = camera.read()
    if not ret:
        print("no camera frame")
        break
    results = model.predict(
        frame,
        imgsz=416,
        conf=0.5,
        verbose=False
    )

    bestresult = {}
    for result in results:
        for box in result.boxes:
            classid = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w_box = x2 - x1
            h_box = y2 - y1
            if w_box < 20 or h_box < 20:
                continue

            if classid not in bestresult or conf > bestresult[classid][0]:
                bestresult[classid] = (conf, box)

    for classid, (conf, box) in bestresult.items():
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{model.names[classid]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Jackblack", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
