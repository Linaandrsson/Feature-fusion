import numpy as np
from pathlib import Path

# Test subject-based splitting logic
base_data_dir = Path('/Volumes/NO NAME/Master Lina/Code/data/Tremor_datagenerator_files')
tremor_variants = ['s2_w2_fs50_tremor_clean', 's2_w2_fs50_tremor_mild_mod', 's2_w2_fs50_tremor_mod_severe']
embeddings_folder_name = 'Activity_ExtractedFeatures'
sensor_name = 'Mag_ankle'

TEST_SUBJECTS = [5, 10]
VAL_SUBJECTS = [2, 7]

all_embeddings = []
all_activities = []
all_subjects = []

print("Loading embeddings from all 3 variants...")
for variant_name in tremor_variants:
    variant_dir = base_data_dir / variant_name
    feat_dir = variant_dir / embeddings_folder_name
    npz_path = feat_dir / f'{sensor_name}_embeddings.npz'
    
    data = np.load(npz_path)
    
    variant_embeddings = np.concatenate([
        data['train_embeddings'],
        data['val_embeddings'],
        data['test_embeddings']
    ], axis=0)
    
    variant_activities = np.concatenate([
        data['train_activities'],
        data['val_activities'],
        data['test_activities']
    ], axis=0)
    
    variant_subjects = np.concatenate([
        data['train_subjects'],
        data['val_subjects'],
        data['test_subjects']
    ], axis=0)
    
    all_embeddings.append(variant_embeddings)
    all_activities.append(variant_activities)
    all_subjects.append(variant_subjects)
    
    print(f"  {variant_name}: {variant_embeddings.shape[0]} samples")

print("\nConcatenating across variants...")
Z_all = np.concatenate(all_embeddings, axis=0)
y_all = np.concatenate(all_activities, axis=0)
subj_all = np.concatenate(all_subjects, axis=0)

print(f"Total samples after concatenation: {len(Z_all)}")

print("\nApplying subject-based split masks...")
train_mask = ~np.isin(subj_all, TEST_SUBJECTS + VAL_SUBJECTS)
val_mask = np.isin(subj_all, VAL_SUBJECTS)
test_mask = np.isin(subj_all, TEST_SUBJECTS)

print(f'\nRESULTS:')
print(f'Train: {np.sum(train_mask)} samples, subjects: {list(np.unique(subj_all[train_mask]))}')
print(f'Val:   {np.sum(val_mask)} samples, subjects: {list(np.unique(subj_all[val_mask]))}')
print(f'Test:  {np.sum(test_mask)} samples, subjects: {list(np.unique(subj_all[test_mask]))}')

print(f'\nEXPECTED:')
print(f'  TRAIN: [1, 3, 4, 6, 8, 9]')
print(f'  VAL:   [2, 7]')
print(f'  TEST:  [5, 10]')

# Verify
expected_train = [1, 3, 4, 6, 8, 9]
expected_val = [2, 7]
expected_test = [5, 10]

train_subjects = list(np.unique(subj_all[train_mask]))
val_subjects = list(np.unique(subj_all[val_mask]))
test_subjects = list(np.unique(subj_all[test_mask]))

if train_subjects == expected_train and val_subjects == expected_val and test_subjects == expected_test:
    print("\n✅ SUBJECT-BASED SPLIT CORRECT!")
else:
    print("\n❌ SUBJECT-BASED SPLIT MISMATCH!")
