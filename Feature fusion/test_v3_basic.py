"""
Quick test of fusion_concat_train_v3.py with minimal config
"""

import sys
sys.path.insert(0, "/Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code/Feature fusion")

# Patch config before import
import fusion_concat_train_v3 as v3

# Override config for quick test
v3.sensors = ["Acc_ankle", "Acc_arm"]
v3.sensor_fs = [10, 10]
v3.epochs = 3  # Just 3 epochs for testing
v3.patience = 2

# Test 1: All clean
print("\n" + "="*70)
print("TEST 1: All sensors clean")
print("="*70)
v3.train_corruption_config = {}
v3.test_corruption_config = {}
v3.main()

# Test 2: Global corruption
print("\n" + "="*70)
print("TEST 2: Global AWGN corruption (30% on all sensors)")
print("="*70)
v3.train_corruption_config = {"AWGN_s0p3": 0.3}
v3.test_corruption_config = {}
v3.main()

# Test 3: Per-sensor corruption
print("\n" + "="*70)
print("TEST 3: Per-sensor corruption")
print("="*70)
v3.train_corruption_config = {
    "AWGN_s0p3/Acc_ankle": 0.2,
    "WEAK_SIGNAL_w0p2/Acc_arm": 0.15,
}
v3.test_corruption_config = {
    "AWGN_s0p3/Acc_ankle": 0.5,
}
v3.main()

print("\n" + "="*70)
print("All tests completed!")
print("="*70)
