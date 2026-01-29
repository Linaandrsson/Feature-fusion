import numpy as np

a = np.load("/Users/linaandersson/Desktop/master/Code/Feature extraction CNNs/Feature extractors/fs50_s2_w2_aug2/ExtractedFeatures/Acc_ankle_embeddings.npz")
b = np.load("/Users/linaandersson/Desktop/master/Code/Feature extraction CNNs/Feature extractors/fs50_s2_w2_aug2/ExtractedFeatures/Acc_chest_embeddings.npz")

print(np.array_equal(a["y_train"], b["y_train"]))
print(np.array_equal(a["y_val"],   b["y_val"])) 
print(np.array_equal(a["y_test"],  b["y_test"]))
