#!/usr/bin/env python3
"""Run check_server.sh and print output — press Run in VS Code to use."""
import subprocess
from pathlib import Path

script = Path(__file__).parent / "check_server.sh"
subprocess.run(["bash", str(script)], check=False)
