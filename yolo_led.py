from ultralytics import YOLO
from gpiozero import LED
import cv2
import threading
import time

MODEL_PATH = "yolo11n.pt"
TARGET_CLASS = "bottle"
TARGET_CONF = 0.70
DISPLAY_CONF = 0.35
IMAGE_SIZE = 416

GREEN_GPIO = 17
RED_GPIO = 27
CAMERA_SOURCE = 0

class LatestFrameCamera:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if isinstance(source, int):
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
            else:
                time.sleep(0.01)

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
green_led.off()
red_led.off()

model = YOLO(MODEL_PATH)
camera = LatestFrameCamera(CAMERA_SOURCE)

if not camera.isOpened():
    raise RuntimeError("Kamera gagal dibuka")

prev_state = None
prev_time = time.time()

try:
    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

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

                if class_name == TARGET_CLASS and score >= TARGET_CONF:
                    target_found = True
                    break

        if target_found != prev_state:
            if target_found:
                green_led.on()
                red_led.off()
            else:
                green_led.off()
                red_led.on()
            prev_state = target_found

        if target_found:
            status = "TARGET TERDETEKSI"
            color = (0, 255, 0)
        else:
            status = "TARGET TIDAK TERDETEKSI"
            color = (0, 0, 255)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-5)
        prev_time = now

        output = result.plot()
        cv2.putText(output, f"{status} | FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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
