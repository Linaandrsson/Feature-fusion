import numpy as np

a = np.load("/Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code/data/Datagenerator_files/fs50_s1_w1_aug2_N30/ExtractedFeatures/Acc_ankle_embeddings_NOISY.npz")
b = np.load("/Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code/data/Datagenerator_files/fs50_s1_w1_aug2_N30/ExtractedFeatures/Acc_chest_embeddings_NOISY.npz")

print(np.array_equal(a["y_train"], b["y_train"]))
print(np.array_equal(a["y_val"],   b["y_val"])) 
print(np.array_equal(a["y_test"],  b["y_test"]))
