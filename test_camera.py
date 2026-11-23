from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Kamera USB tidak dapat dibuka")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=416, conf=0.35, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
EOF

cat << 'EOF' > yolo_led.py
from ultralytics import YOLO
from gpiozero import LED
import cv2
import threading

MODEL_PATH = "yolo11n.pt"
TARGET_CLASS = "bottle"
CONFIDENCE = 0.50
DISPLAY_CONF = 0.35
IMAGE_SIZE = 416

GREEN_GPIO = 17
RED_GPIO = 27
CAMERA_INDEX = 0

class LatestFrameCamera:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()

green_led = LED(GREEN_GPIO)
red_led = LED(RED_GPIO)

model = YOLO(MODEL_PATH)
camera = LatestFrameCamera(CAMERA_INDEX)

if not camera.isOpened():
    green_led.off()
    red_led.off()
    raise RuntimeError("Kamera USB gagal dibuka")

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        result = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=DISPLAY_CONF,
            verbose=False
        )[0]

        target_found = False

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                class_name = result.names[class_id]

                if class_name == TARGET_CLASS and score >= CONFIDENCE:
                    target_found = True
                    break

        if target_found:
            green_led.on()
            red_led.off()
            status = "TARGET TERDETEKSI"
            color = (0, 255, 0)
        else:
            green_led.off()
            red_led.on()
            status = "TARGET TIDAK TERDETEKSI"
            color = (0, 0, 255)

        output = result.plot()
        cv2.putText(output, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("YOLO Raspberry Pi", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    camera.release()
    cv2.destroyAllWindows()
    green_led.off()
    red_led.off()
    green_led.close()
    red_led.close()
EOF
