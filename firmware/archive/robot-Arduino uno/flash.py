import subprocess, sys, os
from pathlib import Path

PIO = Path.home() / ".platformio/penv/Scripts/pio.exe"

os.chdir(os.path.dirname(os.path.abspath(__file__)))
result = subprocess.run([str(PIO), "run", "-e", "usb", "-t", "upload"])
sys.exit(result.returncode)
