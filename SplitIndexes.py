import numpy as np
from sklearn.model_selection import train_test_split

base_dir = "/Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code/data/Datagenerator_files/fs50_s1_w1_aug2_N50" 

ref = np.load(base_dir + "/Acc_ankle.npz")   # X, y, subject_id
y = ref["y"]                         # shape (N,)
subj = ref["subject_id"]             # shape (N,)  (hvis du har)

N = len(y)
all_idx = np.arange(N)

# train vs temp
train_idx, temp_idx = train_test_split(
    all_idx, test_size=0.30, stratify=y, random_state=42
)

# val vs test (50/50 av temp = 15%/15%)
y_temp = y[temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, stratify=y_temp, random_state=42
)

print(len(train_idx), len(val_idx), len(test_idx))

out_path = base_dir + "/splits.npz"

np.savez(
    out_path,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
)

print("Saved splits to:", out_path)

#base_dir = "/Users/linaandersson/Desktop/master/Code/data/Datagenerator_files/fs50_s0.5_w2_aug2"

np.savetxt(f"{base_dir}/train_idx.txt", train_idx, fmt="%d")
np.savetxt(f"{base_dir}/val_idx.txt", val_idx, fmt="%d")
np.savetxt(f"{base_dir}/test_idx.txt", test_idx, fmt="%d")

print("Saved train_idx.txt, val_idx.txt, test_idx.txt")

