import numpy as np

a = np.load("/Users/linaandersson/Desktop/master/Code/Feature extraction CNNs v2/Feature extractors/ExtractedFeatures/Acc_ankle_embeddings.npz")
b = np.load("/Users/linaandersson/Desktop/master/Code/Feature extraction CNNs v2/Feature extractors/ExtractedFeatures/Mag_ankle_embeddings.npz")

print(np.array_equal(a["y_train"], b["y_train"]))
print(np.array_equal(a["y_val"],   b["y_val"])) 
print(np.array_equal(a["y_test"],  b["y_test"]))
