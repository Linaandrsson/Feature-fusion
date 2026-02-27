"""
Test: Verify that parameter file documentation is correct
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import Tremor

# Test write_tremor_parkinson_params_file
output_path = "/tmp/test_tremor_params.txt"

Tremor.write_tremor_parkinson_params_file(
    output_path=output_path,
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

print("=" * 80)
print("PARAMETER FILE OUTPUT:")
print("=" * 80)

with open(output_path, 'r') as f:
    content = f.read()
    print(content)

print("\n" + "=" * 80)
print("✓ Check that mu description is updated:")
print("  - Should say: 'controls nonlinearity / limit-cycle dynamics'")
print("  - Should NOT say: 'controls oscillation frequency'")
print("\n✓ Check that frequency info is included:")
print("  - Should show: 'Frequency: subject-specific (freq_hz, omega = 2πf)'")
print("  - Should list all subject frequencies")
print("=" * 80)
