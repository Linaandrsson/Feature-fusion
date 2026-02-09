"""
Extract features for ALL sensors in parallel or sequentially.

Usage:
    python extract_all_sensors.py
    
This will process all sensors defined in config_extraction.py
"""

import subprocess
import sys
from pathlib import Path
from config_extraction import SENSORS
import time


def main():
    """Extract features for all sensors."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    extract_script = script_dir / "extract_features.py"
    
    print(f"\n{'='*60}")
    print(f"Extracting features for {len(SENSORS)} sensors")
    print(f"{'='*60}\n")
    
    success = []
    failed = []
    
    for i, sensor_name in enumerate(SENSORS.keys(), 1):
        print(f"[{i}/{len(SENSORS)}] Processing: {sensor_name}")
        print("-" * 60)
        
        try:
            # Run extraction for this sensor
            result = subprocess.run(
                [sys.executable, str(extract_script), "--sensor", sensor_name],
                check=True,
                capture_output=False,
                text=True
            )
            success.append(sensor_name)
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to extract features for {sensor_name}")
            failed.append(sensor_name)
        
        print()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Success: {len(success)}/{len(SENSORS)}")
    for s in success:
        print(f"  - {s}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(SENSORS)}")
        for s in failed:
            print(f"  - {s}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
