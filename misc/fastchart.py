"""
fastchart.py — live noisy sine wave at maximum refresh rate.

Uses pyqtgraph for GPU-backed rendering (~100 FPS+).
Install once:  pip install pyqtgraph PyQt6
"""

import sys
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

# ── tunables ────────────────────────────────────────────────────────────────
WINDOW_SECONDS = 4.0      # seconds of history visible
SAMPLE_RATE    = 500      # simulated samples/sec fed into the buffer
REFRESH_HZ     = 60       # GUI repaint rate (Hz)
NOISE_AMP      = 0.30     # noise amplitude relative to sine
SINE_FREQ      = 1.5      # Hz
# ─────────────────────────────────────────────────────────────────────────────

BUFFER_LEN = int(WINDOW_SECONDS * SAMPLE_RATE)

# ring buffers
t_buf = np.zeros(BUFFER_LEN)
y_buf = np.zeros(BUFFER_LEN)
head  = 0          # next write index

t_start = time.perf_counter()
t_last  = t_start


def generate_samples(t_prev: float, t_now: float):
    """Return arrays of (t, y) for all samples in [t_prev, t_now]."""
    n = max(1, int((t_now - t_prev) * SAMPLE_RATE))
    ts = np.linspace(t_prev, t_now, n, endpoint=False)
    ys = (np.sin(2 * np.pi * SINE_FREQ * ts)
          + NOISE_AMP * np.random.randn(n))
    return ts, ys


def update():
    global head, t_last

    t_now = time.perf_counter() - t_start
    ts, ys = generate_samples(t_last, t_now)
    t_last = t_now

    # write into ring buffer
    n = len(ts)
    for i in range(n):
        t_buf[head] = ts[i]
        y_buf[head] = ys[i]
        head = (head + 1) % BUFFER_LEN

    # reconstruct time-ordered view from ring buffer
    idx  = np.arange(head, head + BUFFER_LEN) % BUFFER_LEN
    t_v  = t_buf[idx]
    y_v  = y_buf[idx]

    # keep only the last WINDOW_SECONDS relative to now
    mask = t_v > (t_now - WINDOW_SECONDS)
    curve.setData(t_v[mask] - t_now, y_v[mask])   # x=0 is "now"

    fps_label.setText(f"{1/(time.perf_counter()-t_start - t_last + 1e-9):.0f} fps  "
                      if False else          # skip per-frame calc — use timer below
                      f"t = {t_now:.2f} s")


# ── build UI ─────────────────────────────────────────────────────────────────
app = QtWidgets.QApplication(sys.argv)

pg.setConfigOptions(antialias=False,   # off = faster
                    useOpenGL=True,    # GPU compositing
                    enableExperimental=True)

win = pg.GraphicsLayoutWidget(title="Live Noisy Sine — fastchart.py", show=True)
win.resize(1000, 400)

plot = win.addPlot(title="")
plot.setLabel("bottom", "time (s, 0 = now)")
plot.setLabel("left",   "amplitude")
plot.setYRange(-1.6, 1.6, padding=0)
plot.setXRange(-WINDOW_SECONDS, 0.05, padding=0)
plot.showGrid(x=True, y=True, alpha=0.25)
plot.getAxis("bottom").setTickSpacing(major=1, minor=0.25)

curve = plot.plot(pen=pg.mkPen(color=(80, 200, 120), width=1))

# fps readout in corner
fps_label = pg.LabelItem(justify="right")
win.addItem(fps_label)

# fire update() at REFRESH_HZ
timer = QtCore.QTimer()
timer.setInterval(int(1000 / REFRESH_HZ))
timer.timeout.connect(update)
timer.start()

sys.exit(app.exec())
