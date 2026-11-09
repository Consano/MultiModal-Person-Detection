import os
import re
import struct
import sys
import time

import cv2
import numpy as np
import serial

SERIAL_PORT = "COM11"
BAUD_RATE = 921600
IMG_W, IMG_H = 160, 120
FRAME_BYTES = IMG_W * IMG_H * 2          # both formats are 2 bytes per pixel (38,400)
OUTPUT_DIR = r"C:\Users\jackk\Desktop\SeniorProject"

PIXEL_FORMAT = "rgb565"   # "rgb565" or "yuv422" - must match OUTPUT_YUV422 in the sketch
SWAP_BYTES = False        # rgb565 only: flip if colours look scrambled

# Polarity combos understood by the sketch: bit0 = VSPOL, bit1 = HSPOL, bit2 = PCLK rising.
# For this camera the expected-good combos are 4 and 0 (blanking level low for both syncs).
SCAN_POLARITY = True
DEFAULT_COMBO = 4


def capture(ser, timeout=12.0):
    """Ask for one frame. Returns (status, data)."""
    ser.reset_input_buffer()
    ser.write(b"c")
    buf = bytearray()
    t0 = last_rx = time.time()
    nudges = 0
    while time.time() - t0 < timeout:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf.extend(chunk)
            last_rx = time.time()

        i = buf.find(b"FRAM")
        if i != -1:
            if len(buf) >= i + 5 + FRAME_BYTES:
                overrun = buf[i + 4] == ord("O")
                return ("overrun" if overrun else "ok"), bytes(buf[i + 5 : i + 5 + FRAME_BYTES])
            # Header arrived but the tail is stuck in the USB buffer: ask the board for padding
            if time.time() - last_rx > 1.0 and nudges < 3:
                ser.write(b"p")
                nudges += 1
                last_rx = time.time()

        j = buf.find(b"ERRTO")
        if j != -1 and len(buf) >= j + 9:
            words = struct.unpack("<I", buf[j + 5 : j + 9])[0]
            return "timeout", words
    return "silent", len(buf)


def decode(payload):
    """Returns (grayscale, BGR colour) images."""
    if PIXEL_FORMAT == "yuv422":
        raw = np.frombuffer(payload, dtype=np.uint8).reshape((IMG_H, IMG_W * 2))
        gray = raw[:, 0::2].copy()
        color = cv2.cvtColor(raw.reshape((IMG_H, IMG_W, 2)), cv2.COLOR_YUV2BGR_YUY2)
        return gray, color
    px = np.frombuffer(payload, dtype=">u2" if SWAP_BYTES else "<u2").reshape((IMG_H, IMG_W))
    r = ((px >> 11) & 0x1F).astype(np.uint16) * 255 // 31
    g = ((px >> 5) & 0x3F).astype(np.uint16) * 255 // 63
    b = (px & 0x1F).astype(np.uint16) * 255 // 31
    color = np.dstack([b, g, r]).astype(np.uint8)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    return gray, color


def montage(frames):
    """Contact sheet of all successful polarity combos, labelled."""
    tiles = []
    for combo, color, mean in frames:
        tile = cv2.resize(color, (IMG_W * 2, IMG_H * 2), interpolation=cv2.INTER_NEAREST)
        label = f"combo {combo} VS={combo & 1} HS={(combo >> 1) & 1} PCK={(combo >> 2) & 1}"
        cv2.putText(tile, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(tile, f"mean brightness {mean:.0f}", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k : k + 3]) for k in range(0, len(tiles), 3)]
    return np.vstack(rows)


print(f"Connecting to {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.5)
    ser.setDTR(True)
    ser.setRTS(True)
    time.sleep(2)
except Exception as e:
    print(f"Error opening port: {e}")
    sys.exit(1)

ser.reset_input_buffer()
ser.write(b"i")
print("Camera ID check:", ser.readline().decode(errors="replace").strip() or "(no reply)")
print("  expect: CAMID 26 42 NACKS 0   (some modules report 41)")

# Sweep the sensor clock divider; a setting is "plausible" if it shows a sensible number of
# VSYNC pulses and roughly a frame's worth of lines between them. Fastest -> slowest; the
# slowest plausible one wins (easier for the polling loop to keep up with).
best_clk = None
for clk in ["1", "3", "7", "f"]:
    ser.write(b"k" + clk.encode())
    time.sleep(0.5)
    ser.reset_input_buffer()
    ser.write(b"d")
    ser.timeout = 4
    line = ser.readline().decode(errors="replace").strip()
    ser.timeout = 0.5
    print(f"CLKRC 0x0{clk}: {line or '(no reply)'}")
    m_vs = re.search(r"VSr=(\d+)", line)
    m_ln = re.search(r"lines/frame=(\d+)", line)
    vs = int(m_vs.group(1)) if m_vs else 0
    ln = int(m_ln.group(1)) if m_ln else 0
    if 2 <= vs <= 200 and 60 <= ln <= 400:
        best_clk = clk

if best_clk is None:
    print("WARNING: no clock setting showed a plausible VSYNC pattern - falling back to 0x03.")
    best_clk = "3"
ser.write(b"k" + best_clk.encode())
time.sleep(0.5)
print(f"Using CLKRC 0x0{best_clk} for capture")

combos = [DEFAULT_COMBO]
if SCAN_POLARITY:
    combos += [c for c in [0, 4, 1, 2, 3, 5, 6, 7] if c != DEFAULT_COMBO]

good = []
for combo in combos:
    ser.write(str(combo).encode())
    time.sleep(0.2)
    status, data = capture(ser)
    label = f"combo {combo} (VS={combo & 1}, HS={(combo >> 1) & 1}, PCK={(combo >> 2) & 1})"

    if status in ("ok", "overrun"):
        gray, color = decode(data)
        base = os.path.join(OUTPUT_DIR, f"captured_p{combo}")
        cv2.imwrite(base + "_gray.png", gray)
        cv2.imwrite(base + "_color.png", color)
        note = " (FIFO overrun - try a slower clock)" if status == "overrun" else ""
        print(f"{label}: got a frame{note}, mean brightness {gray.mean():.0f}, saved {base}_*.png")
        good.append((combo, color, float(gray.mean())))
        if not SCAN_POLARITY:
            break
    elif status == "timeout":
        print(f"{label}: DCMI timeout, {data}/{FRAME_BYTES // 4} words received")
    else:
        print(f"{label}: incomplete reply from board ({data} bytes)")

ser.close()

if good:
    sheet = montage(good)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "captured_montage.png"), sheet)
    cv2.imshow("All polarity combos that returned a frame", sheet)
    print("Look for the combo with a recognisable picture. Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No frames received. Check the messages above.")