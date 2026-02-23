from ultralytics import YOLO
import cv2
import os
import time

# map of card values (excluding jokers) to Hi-Lo count values
hi_lo_values = {
    "2": 1, "3": 1, "4": 1, "5": 1, "6": 1,
    "7": 0, "8": 0, "9": 0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1
}

modelpath = os.path.join(os.path.dirname(__file__), "Model", "yolov8m_synthetic.pt")
model = YOLO(modelpath)

# open camera (automatic windows choose index) depending on system
camera = cv2.VideoCapture(cv2.CAP_DSHOW)
cv2.namedWindow("Jackblack", cv2.WINDOW_AUTOSIZE)
if not camera.isOpened():
    print("camera not working")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



# running count state
running_count = 0
counted_cards = set()   # card identities already counted (e.g. 10h)

# main loop
while True:
    ret, frame = camera.read()
    if not ret:
        break

    # yolo inference
    results = model.predict(
        frame,
        imgsz=800, # how big compress image to
        conf=0.5, # confidence threshold. model must be higher than 90% sure its the card to avoid miscount
        half=True,
        device=0,
        verbose=False
    )

    bestresult = {}

    # keep best detection per class
    for result in results:
        for box in result.boxes:
            classid = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            if classid not in bestresult or conf > bestresult[classid][0]:
                bestresult[classid] = (conf, box)

    # draw boxes and update count
    for classid, (conf, box) in bestresult.items():
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        card_name = model.names[classid]   # 10h, As
        rank = card_name[:-1]              # remove suit to get rank

        # count each card identity once per 
        if card_name not in counted_cards:
            counted_cards.add(card_name)
            if rank in hi_lo_values:
                running_count += hi_lo_values[rank]

        label = f"{card_name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )


    # running count display
    cv2.putText(
        frame, f"Running Count: {running_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
    )

    cv2.imshow("Jackblack", frame)

    # controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # q to quit (close window)
        break
    elif key == ord('r'): # r to reset count (set back to 0. clears counted cards)
        running_count = 0
        counted_cards.clear()

# cleanup
camera.release()
cv2.destroyAllWindows()
