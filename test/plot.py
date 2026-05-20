import matplotlib
matplotlib.use('TkAgg')

import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================
# SERIAL CONFIGURATION
# ==========================================

COM_PORT = "COM7"      # CHANGE THIS
BAUD = 115200

# ==========================================
# OPEN SERIAL
# ==========================================

ser = serial.Serial(
    COM_PORT,
    BAUD,
    timeout=1
)

# Wait ESP32 reset
import time
time.sleep(2)

# ==========================================
# SETTINGS
# ==========================================

WINDOW_SIZE = 500
SAMPLE_RATE = 16000

raw_data = []
filtered_data = []

# ==========================================
# CREATE FIGURES
# ==========================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(12,8)
)

# ==========================================
# TIME DOMAIN
# ==========================================

line_raw, = ax1.plot(
    [],
    [],
    linewidth=1,
    label="Before Filter"
)

line_filtered, = ax1.plot(
    [],
    [],
    linewidth=2,
    label="After Filter"
)

ax1.set_title(
    "Real-Time Time Domain"
)

ax1.set_xlabel("Samples")

ax1.set_ylabel("Amplitude")

ax1.set_xlim(0, WINDOW_SIZE)

ax1.set_ylim(-5000, 5000)

ax1.grid(True)

ax1.legend()

# ==========================================
# FFT DOMAIN
# ==========================================

line_fft_raw, = ax2.plot(
    [],
    [],
    linewidth=1,
    label="Raw FFT"
)

line_fft_filtered, = ax2.plot(
    [],
    [],
    linewidth=2,
    label="Filtered FFT"
)

ax2.set_title(
    "Real-Time Frequency Domain"
)

ax2.set_xlabel("Frequency (Hz)")

ax2.set_ylabel("Magnitude")

ax2.set_xlim(0, 8000)

ax2.set_ylim(0, 500000)

ax2.grid(True)

ax2.legend()

# ==========================================
# UPDATE FUNCTION
# ==========================================

def update(frame):

    global raw_data
    global filtered_data

    while ser.in_waiting:

        try:

            line = ser.readline() \
                      .decode("utf-8") \
                      .strip()

            values = line.split(",")

            if len(values) != 2:
                continue

            raw = int(values[0])

            filt = int(values[1])

            raw_data.append(raw)

            filtered_data.append(filt)

            # Keep fixed window size
            if len(raw_data) > WINDOW_SIZE:

                raw_data.pop(0)

                filtered_data.pop(0)

        except:
            pass

    # ======================================
    # UPDATE TIME DOMAIN
    # ======================================

    x_axis = np.arange(len(raw_data))

    line_raw.set_data(
        x_axis,
        raw_data
    )

    line_filtered.set_data(
        x_axis,
        filtered_data
    )

    # ======================================
    # UPDATE FFT
    # ======================================

    if len(raw_data) > 64:

        raw_array = np.array(raw_data)

        filt_array = np.array(filtered_data)

        fft_raw = np.abs(
            np.fft.rfft(raw_array)
        )

        fft_filt = np.abs(
            np.fft.rfft(filt_array)
        )

        freqs = np.fft.rfftfreq(
            len(raw_array),
            d=1/SAMPLE_RATE
        )

        line_fft_raw.set_data(
            freqs,
            fft_raw
        )

        line_fft_filtered.set_data(
            freqs,
            fft_filt
        )

    return (
        line_raw,
        line_filtered,
        line_fft_raw,
        line_fft_filtered
    )

# ==========================================
# ANIMATION
# ==========================================

ani = FuncAnimation(
    fig,
    update,
    interval=10,
    blit=False,
    cache_frame_data=False
)

plt.tight_layout()

print("Real-time plot started...")

plt.show()

# ==========================================
# CLOSE SERIAL
# ==========================================

ser.close()