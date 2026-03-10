import numpy as np

# Load split indexes
train_idx = np.loadtxt('data/Tremor_datagenerator_files/s2_w2_fs50_tremor_clean/splits/train_idx.txt', dtype=int)
val_idx = np.loadtxt('data/Tremor_datagenerator_files/s2_w2_fs50_tremor_clean/splits/val_idx.txt', dtype=int)
test_idx = np.loadtxt('data/Tremor_datagenerator_files/s2_w2_fs50_tremor_clean/splits/test_idx.txt', dtype=int)

print('Index file splits:')
print(f'  Train: {len(train_idx)} samples, range [{train_idx.min()}, {train_idx.max()}]')  
print(f'  Val: {len(val_idx)} samples, range [{val_idx.min()}, {val_idx.max()}]')
print(f'  Test: {len(test_idx)} samples, range [{test_idx.min()}, {test_idx.max()}]')

# Load actual data to check subjects
data = np.loadtxt('data/Tremor_datagenerator_files/s2_w2_fs50_tremor_clean/Mag_ankle.txt', delimiter=',')
col_offset = 3 * 100
y_subject = data[:, col_offset + 1].astype(np.int64)

print('\nSubjects in each split (using index files):')
print(f'  Train subjects: {np.unique(y_subject[train_idx])}')
print(f'  Val subjects: {np.unique(y_subject[val_idx])}')
print(f'  Test subjects: {np.unique(y_subject[test_idx])}')

# Check if there's overlap
train_set = set(np.unique(y_subject[train_idx]))
val_set = set(np.unique(y_subject[val_idx]))
test_set = set(np.unique(y_subject[test_idx]))

print('\nOverlap check:')
print(f'  Train ∩ Val: {train_set & val_set}')
print(f'  Train ∩ Test: {train_set & test_set}')
print(f'  Val ∩ Test: {val_set & test_set}')

print('\nExpected subject-based split:')
print('  TEST: [5, 10]')
print('  VAL: [2, 7]')
print('  TRAIN: [1, 3, 4, 6, 8, 9]')
