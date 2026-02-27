#!/usr/bin/env python3
"""
Test script to verify the updated Tremor.py functions work correctly
with the new A_SUBJECT format.
"""
import sys
sys.path.insert(0, '/Volumes/NO NAME/Master Lina/Code')
from Noise_simulation.Tremor import write_tremor_parkinson_params_file
import tempfile
import os

print("Testing write_tremor_parkinson_params_file()...")
print("=" * 70)

# Create a temporary file
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
    temp_path = f.name

try:
    # Call the function
    write_tremor_parkinson_params_file(
        output_path=temp_path,
        mu=1.0,
        sigma=0.5,
        dt=0.001,
        intermittent=False,
        on_prob=0.5,
        min_on_sec=2.0,
        max_on_sec=8.0,
        use_jitter=True,
        jitter_std=0.15,
        scenario_seed=42
    )
    
    # Read and display the file
    with open(temp_path, 'r') as f:
        content = f.read()
        print(content)
    
    print("\n" + "=" * 70)
    print("✓ write_tremor_parkinson_params_file() succeeded!")
    print("✓ No crashes from accessing params['acc']")
    
finally:
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
