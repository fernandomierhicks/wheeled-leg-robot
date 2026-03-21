"""viz — pyqtgraph visualization for master_sim.

Imports are lazy to avoid loading pyqtgraph at package-init time.
On Windows, mujoco must be imported before pyqtgraph (OpenGL DLL conflict),
so eager imports here would break any module that imports mujoco after viz.
"""


def __getattr__(name):
    if name in ("show", "load_csv", "ChartPanel", "replay", "replay_show",
                "TelemetryRecorder"):
        from . import visualizer
        return getattr(visualizer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
