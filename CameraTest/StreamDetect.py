"""Live video from the Portenta H7 + OV2640 with person detection.

Keys in the video window:  q = quit,  d = toggle detection,  s = save snapshot
"""
import collections
import csv
import os
import struct
import sys
import threading
import time

import cv2
import numpy as np
import serial

# ----------------------------------------------------------------------------- settings
SERIAL_PORT = "COM11"
BAUD_RATE = 921600
SNAPSHOT_DIR = r"C:\Users\jackk\Desktop\SeniorProject"

def save_snapshot(img):
    """Save the annotated view as a PNG. Falls back to the script folder if SNAPSHOT_DIR is unusable."""
    name = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
    for folder in (SNAPSHOT_DIR, os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        try:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, name)
            if cv2.imwrite(path, img):
                print("Saved snapshot:", path)
                return path
        except Exception as e:
            print(f"Could not save to {folder}: {e}")
    print("Snapshot FAILED in every folder tried.")
    return None


POLARITY_COMBO = 4      # 4 = VS low, HS low, PCLK rising (validated on your hardware)
CLKRC = 0x03            # sensor clock divider: 0x00 = fastest. DMA stalled at 0x00 on this board, so start at 0x03
                        # (polling mode ignores anything faster than 0x07)
DVP_DIV = 0x08          # pixel-clock divider (reg 0xD3); lower = faster, ST default 0x08

LOG_CSV = True          # write one row per frame (timings + scores) to stream_log_<time>.csv in SNAPSHOT_DIR
PC_DETECTOR = False     # True = also run a PC-side person detector (boxes) for comparison with the board
PRESENCE_THRESHOLD = 60 # smoothed person score (0-100) at/above which someone counts as present
SMOOTH_N = 3            # number of recent frames averaged for the presence decision
USE_YOLO = False        # True needs: pip install ultralytics (downloads yolov8n.pt on first use)
DETECT_UPSCALE = 2      # HOG only: upscale before detecting (HOG looks for people >= 128 px tall)
MIN_SCORE = 0.3         # HOG weight threshold
DISPLAY_SCALE = 2

# ----------------------------------------------------------------------------- protocol
HEADER = struct.Struct("<4sIHHBBIBBHHHH")  # magic, counter, w, h, format, flags, length, person%, notperson%, infer_ms, cap_ms, prep_ms, send_ms (28 bytes)
MAGIC = b"FRMX"


class FrameParser:
    """Turns the raw serial byte stream into frames; resynchronises on the magic bytes."""

    def __init__(self):
        self.buf = bytearray()
        self.skipped = 0

    def feed(self, data):
        self.buf.extend(data)
        frames = []
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                if len(self.buf) > 3:               # keep a possible partial magic at the end
                    self.skipped += len(self.buf) - 3
                    del self.buf[:-3]
                break
            if i > 0:
                self.skipped += i
                del self.buf[:i]
            if len(self.buf) < HEADER.size:
                break
            (_, counter, w, h, fmt, flags, length, p_person, p_notp, infer_ms,
             cap_ms, prep_ms, send_ms) = HEADER.unpack_from(self.buf)
            valid = (0 < w <= 1024 and 0 < h <= 1024 and fmt in (0, 1)
                     and length == w * h * (1 if fmt == 0 else 2))
            if not valid:                           # false magic inside pixel data: skip a byte
                del self.buf[:1]
                self.skipped += 1
                continue
            if len(self.buf) < HEADER.size + length:
                break
            payload = bytes(self.buf[HEADER.size:HEADER.size + length])
            del self.buf[:HEADER.size + length]
            frames.append((counter, w, h, fmt, flags, payload, p_person, p_notp, infer_ms,
                           cap_ms, prep_ms, send_ms))
        return frames


def to_images(w, h, fmt, payload):
    """Returns (gray, bgr)."""
    if fmt == 0:
        gray = np.frombuffer(payload, dtype=np.uint8).reshape((h, w))
        return gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raw = np.frombuffer(payload, dtype=np.uint8).reshape((h, w, 2))
    bgr = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUY2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr


class Receiver(threading.Thread):
    """Drains the serial port continuously and keeps only the newest frame, so a slow
    detector never makes the video lag behind (old frames are simply dropped)."""

    def __init__(self, ser):
        super().__init__(daemon=True)
        self.ser = ser
        self.parser = FrameParser()
        self.lock = threading.Lock()
        self.latest = None
        self.received = 0
        self.overruns = 0
        self.running = True

    def run(self):
        while self.running:
            try:
                data = self.ser.read(max(1, self.ser.in_waiting))
            except serial.SerialException:
                break
            if not data:
                continue
            for frame in self.parser.feed(data):
                with self.lock:
                    self.latest = frame
                    self.received += 1
                    if frame[4] & 1:
                        self.overruns += 1

    def take(self, last_counter):
        with self.lock:
            if self.latest is not None and self.latest[0] != last_counter:
                return self.latest
        return None


# ----------------------------------------------------------------------------- detection
def make_detector():
    """Returns detect(gray, bgr) -> [(x, y, w, h, score), ...], or None if nothing is available."""
    want_yolo = USE_YOLO
    if not want_yolo and not hasattr(cv2, "HOGDescriptor"):
        print("This OpenCV build (%s) has no HOGDescriptor (removed in OpenCV 5); trying YOLO instead."
              % cv2.__version__)
        want_yolo = True

    if want_yolo:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("No person detector available. Either 'pip install ultralytics' (YOLO), or install "
                  "OpenCV 4 ('pip install \"opencv-python<5\"') to use the built-in HOG detector.\n"
                  "Video will still be shown, without detection.")
            return None
        model = YOLO("yolov8n.pt")

        def detect_yolo(gray, bgr):
            r = model(bgr, imgsz=320, conf=0.35, classes=[0], verbose=False)[0]
            out = []
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(c)))
            return out
        return detect_yolo

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_hog(gray, bgr):
        s = DETECT_UPSCALE
        img = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR) if s != 1 else gray
        rects, weights = hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
        boxes, scores = [], []
        for (x, y, w, h), wt in zip(rects, np.ravel(weights)):
            if wt >= MIN_SCORE:
                boxes.append([int(x / s), int(y / s), int(w / s), int(h / s)])
                scores.append(float(wt))
        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxes(boxes, scores, MIN_SCORE, 0.4)
        return [tuple(boxes[i]) + (scores[i],) for i in np.ravel(keep)]
    return detect_hog


# ----------------------------------------------------------------------------- main
def main():
    print(f"Connecting to {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        try:
            ser.set_buffer_size(rx_size=1 << 20)
        except Exception:
            pass
        ser.setDTR(True)
        ser.setRTS(True)
        time.sleep(2)
    except Exception as e:
        print(f"Error opening port: {e}")
        sys.exit(1)

    ser.reset_input_buffer()
    ser.write(b"x")                                  # make sure nothing is streaming yet
    time.sleep(0.3)
    ser.reset_input_buffer()
    ser.write(b"i")
    ser.timeout = 1.0
    print("Board info:", ser.readline().decode(errors="replace").strip() or "(no reply)")
    ser.timeout = 0.1

    ser.write(b"r")                                  # retry DMA capture (board may be in polling fallback)
    ser.write(str(POLARITY_COMBO).encode())
    ser.write(b"k%02x" % CLKRC)
    ser.write(b"v%02x" % DVP_DIV)
    time.sleep(0.3)
    ser.reset_input_buffer()

    receiver = Receiver(ser)
    receiver.start()
    ser.write(b"s")                                  # start streaming

    detect = make_detector() if PC_DETECTOR else None
    recent = collections.deque(maxlen=SMOOTH_N)
    present = None
    detecting = detect is not None
    last_counter = None
    fps_t0, fps_n, fps = time.time(), 0, 0.0
    det_ms = 0.0
    people = []
    shown = 0
    t_start = time.time()
    last_log = t_start
    last_frame_t = None
    last_logged = None
    last_view = None
    smooth = 0.0
    stats = {k: [] for k in ('cap', 'prep', 'infer', 'send', 'period')}
    log_file, log_writer = None, None
    if LOG_CSV:
        try:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            log_file = open(os.path.join(SNAPSHOT_DIR, f"stream_log_{int(t_start)}.csv"), "w", newline="")
            log_writer = csv.writer(log_file)
            log_writer.writerow(["t_s", "frame", "person_pct", "notperson_pct", "smoothed", "present", "model_ok",
                                 "cap_ms", "prep_ms", "infer_ms", "send_ms", "host_period_ms", "mean_brightness", "overrun"])
        except Exception as e:
            print(f"CSV logging disabled ({e})")

    print("Streaming. Keys: q = quit, d = toggle detection, s = save snapshot")
    while True:
        frame = receiver.take(last_counter)
        if frame is None:
            if time.time() - t_start > 15 and receiver.received == 0:
                print("No frames after 15 s. Check the board info line above (dma/timeouts) and wiring.")
                break
            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and last_view is not None:
                save_snapshot(last_view)
            if key == ord("d") and detect is not None:
                detecting = not detecting
            continue

        counter, w, h, fmt, flags, payload, p_person, p_notp, infer_ms, cap_ms, prep_ms, send_ms = frame
        now = time.time()
        host_dt = (now - last_frame_t) if last_frame_t else 0.0
        last_frame_t = now
        last_counter = counter
        gray, bgr = to_images(w, h, fmt, payload)

        if detecting:
            t0 = time.time()
            people = detect(gray, bgr)
            det_ms = (time.time() - t0) * 1000
        else:
            people = []

        shown += 1
        fps_n += 1
        if time.time() - fps_t0 >= 1.0:
            fps = fps_n / (time.time() - fps_t0)
            fps_n, fps_t0 = 0, time.time()

        if time.time() - last_log >= 5:
            print(f"[{time.time() - t_start:5.0f}s] {fps:.2f} fps | frame #{counter} | mean {gray.mean():.0f} | "
                  f"board ms: cap {cap_ms} prep {prep_ms} infer {infer_ms} send {send_ms}")
            last_log = time.time()

        view = cv2.resize(bgr, (w * DISPLAY_SCALE, h * DISPLAY_SCALE), interpolation=cv2.INTER_NEAREST)
        for (x, y, bw, bh, score) in people:
            p1 = (x * DISPLAY_SCALE, y * DISPLAY_SCALE)
            p2 = ((x + bw) * DISPLAY_SCALE, (y + bh) * DISPLAY_SCALE)
            cv2.rectangle(view, p1, p2, (0, 255, 0), 2)
            cv2.putText(view, f"{score:.2f}", (p1[0], max(12, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        model_ok = bool(flags & 2)
        status = (f"{fps:.1f} fps | #{counter} mean {gray.mean():.0f}"
                  + (f" | board model {infer_ms} ms" if model_ok else " | board model OFF")
                  + (f" | PC people: {len(people)}" if detecting else "")
                  + (" | OVERRUN" if flags & 1 else ""))
        cv2.putText(view, status, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if model_ok:
            recent.append(p_person)
            smooth = sum(recent) / len(recent)
            now_present = smooth >= PRESENCE_THRESHOLD
            if now_present != present:
                present = now_present
                print(f"[{time.time() - t_start:5.0f}s] {'PERSON PRESENT' if present else 'no person'} "
                      f"(score {smooth:.0f}, frame #{counter})")
            color = (0, 255, 0) if present else (180, 180, 180)
            label = f"PERSON PRESENT  {smooth:.0f}%" if present else f"no person  {smooth:.0f}%"
            cv2.putText(view, label, (10, view.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(view, f"raw: person {p_person}%  not-person {p_notp}%", (10, view.shape[0] - 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(view, "board classifier not running (check 'model=' in the board info line)",
                        (10, view.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # inset: roughly what the 96x96 model sees (centre square, scaled down)
        side = min(w, h)
        x0, y0 = (w - side) // 2, (h - side) // 2
        inset = cv2.resize(gray[y0:y0 + side, x0:x0 + side], (96, 96), interpolation=cv2.INTER_AREA)
        inset = cv2.cvtColor(cv2.resize(inset, (192, 192), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        if view.shape[0] > 220 and view.shape[1] > 220:
            view[28:28 + 192, view.shape[1] - 202:view.shape[1] - 10] = inset
        if counter > 1 and host_dt > 0 and counter != last_logged:
            stats["cap"].append(cap_ms); stats["prep"].append(prep_ms)
            stats["infer"].append(infer_ms); stats["send"].append(send_ms); stats["period"].append(host_dt * 1000)
        if log_writer is not None and counter != last_logged:
            log_writer.writerow([f"{now - t_start:.2f}", counter, p_person, p_notp,
                                 f"{smooth:.1f}" if model_ok else "", int(bool(present)) if model_ok else "",
                                 int(model_ok), cap_ms, prep_ms, infer_ms, send_ms, f"{host_dt * 1000:.0f}",
                                 f"{gray.mean():.0f}", flags & 1])
            if counter % 20 == 0:
                log_file.flush()
        last_logged = counter
        last_view = view.copy()
        cv2.imshow("Portenta OV2640 stream", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d") and detect is not None:
            detecting = not detecting
        if key == ord("s"):
            save_snapshot(last_view)

    receiver.running = False
    receiver.join(timeout=1.0)
    ser.write(b"x")
    time.sleep(0.5)
    ser.reset_input_buffer()
    ser.write(b"i")
    t_end, tail = time.time() + 2.0, bytearray()
    ser.timeout = 0.2
    while time.time() < t_end and b"CAMID" not in tail:
        tail.extend(ser.read(4096))
    k = tail.find(b"CAMID")
    print("Board info after run:",
          tail[k:].split(b"\n")[0].decode(errors="replace").strip() if k >= 0 else "(no reply)")
    ser.close()
    cv2.destroyAllWindows()
    def _summ(name, vals):
        if vals:
            print(f"  {name:<14} mean {sum(vals) / len(vals):7.1f} ms | max {max(vals):6.0f} | min {min(vals):6.0f}")
    print("Timing summary (per frame):")
    _summ("board capture", stats["cap"]); _summ("board prep", stats["prep"])
    _summ("board infer", stats["infer"]); _summ("board send", stats["send"])
    _summ("frame period", stats["period"])
    if log_file is not None:
        log_file.close()
        print("Per-frame log saved:", log_file.name)
    print(f"Done. Frames received: {receiver.received}, overruns flagged: {receiver.overruns}, "
          f"bytes skipped while resyncing: {receiver.parser.skipped}")


if __name__ == "__main__":
    main()
