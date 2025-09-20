import subprocess
import sys

def install_runtime_packages():
    packages = [
        "torch==2.2.2",
        "transformers==4.56.2"
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", package])

try:
    import torch
    import transformers
except ImportError:
    install_runtime_packages()
