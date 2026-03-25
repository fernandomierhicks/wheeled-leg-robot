"""Allow `python -m master_sim.viz` to launch the visualizer."""
import multiprocessing
multiprocessing.freeze_support()

from .visualizer import main

main()
