import numpy as np
from pathlib import Path

# Load raw tremor data
data_file = Path('data/Tremor_datagenerator_files/s2_w2_fs50_tremor_clean/Mag_ankle.txt')
data = np.loadtxt(data_file, delimiter=',')

# Extract subjects (column 101 for 3-channel sensors)
col_offset = 3 * 100
y_subject = data[:, col_offset + 1].astype(np.int64)

print(f'Total samples: {len(y_subject)}')
print(f'Unique subjects in raw data: {np.unique(y_subject)}')
print(f'Samples per subject:')
for subj in np.unique(y_subject):
    count = np.sum(y_subject == subj)
    print(f'  Subject {subj}: {count} samples')

# Test split logic
VAL_SUBJECTS = [2, 7]
TEST_SUBJECTS = [5, 10]

print(f'\nSplit configuration:')
print(f'  VAL_SUBJECTS: {VAL_SUBJECTS}')
print(f'  TEST_SUBJECTS: {TEST_SUBJECTS}')
print(f'  VAL + TEST: {VAL_SUBJECTS + TEST_SUBJECTS}')

train_mask = ~np.isin(y_subject, VAL_SUBJECTS + TEST_SUBJECTS)
val_mask = np.isin(y_subject, VAL_SUBJECTS)
test_mask = np.isin(y_subject, TEST_SUBJECTS)

print(f'\nAfter split:')
print(f'  Train mask True count: {np.sum(train_mask)} (subjects: {np.unique(y_subject[train_mask])})')
print(f'  Val mask True count: {np.sum(val_mask)} (subjects: {np.unique(y_subject[val_mask])})')
print(f'  Test mask True count: {np.sum(test_mask)} (subjects: {np.unique(y_subject[test_mask])})')

# Now load the actual embedding file and check
embedding_file = Path('data/Tremor_datagenerator_files/s2_w2_fs50_tremor_clean/Activity_ExtractedFeatures/Mag_ankle_embeddings.npz')
if embedding_file.exists():
    print(f'\n\nChecking saved embedding file...')
    npz = np.load(embedding_file)
    print(f'Keys in NPZ: {list(npz.keys())}')
    print(f'Train subjects in NPZ: {np.unique(npz["train_subjects"])}')
    print(f'Val subjects in NPZ: {np.unique(npz["val_subjects"])}')
    print(f'Test subjects in NPZ: {np.unique(npz["test_subjects"])}')
