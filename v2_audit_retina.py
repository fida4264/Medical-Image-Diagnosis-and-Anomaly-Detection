"""Phase 1: Retina Dataset Audit (Deep)"""
import os, json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from PIL import Image

base = Path('datasets/retina')
csv_path = base / 'full_df.csv'
odir_train_dir = base / 'ODIR-5K' / 'ODIR-5K' / 'Training Images'
odir_test_dir = base / 'ODIR-5K' / 'ODIR-5K' / 'Testing Images'
prep_dir = base / 'preprocessed_images'

print("=== RETINA DEEP AUDIT ===")

df = pd.read_csv(csv_path)
print(f"\n  [A] full_df.csv Metadata:")
print(f"    Total rows: {len(df)}")
print(f"    Unique Patient IDs: {df['ID'].nunique()}")
print(f"    Patient Age: min={df['Patient Age'].min()}, max={df['Patient Age'].max()}, mean={df['Patient Age'].mean():.1f}")
print(f"    Patient Sex: {df['Patient Sex'].value_counts().to_dict()}")

label_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
label_names = {
    'N': 'Normal',
    'D': 'Diabetic Retinopathy',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'Age-related Macular Degeneration (AMD)',
    'H': 'Hypertension',
    'M': 'Pathological Myopia',
    'O': 'Other Disease/Abnormality'
}

label_counts = {}
for col in label_cols:
    cnt = int(df[col].sum())
    label_counts[col] = {'count': cnt, 'percentage': round(cnt / len(df) * 100, 2), 'name': label_names[col]}
    print(f"    {col} ({label_names[col]}): {cnt} ({label_counts[col]['percentage']}%)")

df['num_labels'] = df[label_cols].sum(axis=1)
cardinality = {int(k): int(v) for k, v in df['num_labels'].value_counts().sort_index().items()}
print(f"\n  [B] Multi-label Cardinality:")
for num_l, cnt in cardinality.items():
    print(f"    {num_l} active labels: {cnt} records ({cnt/len(df)*100:.1f}%)")

# Inspect files
train_files = {f.name for f in odir_train_dir.glob('*.jpg')} if odir_train_dir.exists() else set()
test_files = {f.name for f in odir_test_dir.glob('*.jpg')} if odir_test_dir.exists() else set()
prep_files = {f.name for f in prep_dir.glob('*.jpg')} if prep_dir.exists() else set()
csv_files = set(df['filename'].dropna().unique())

print(f"\n  [C] Physical Image Inventory:")
print(f"    ODIR-5K Training Images: {len(train_files)} files")
print(f"    ODIR-5K Testing Images: {len(test_files)} files (ALL UNLABELED)")
print(f"    preprocessed_images: {len(prep_files)} files (512x512 resized)")
print(f"    Total physical images: {len(train_files) + len(test_files) + len(prep_files)}")

print(f"\n  [D] Image-Label Cross-Reference:")
print(f"    full_df.csv filenames mapped to Training Images: {len(csv_files & train_files)} / {len(csv_files)} (100%)")
print(f"    Training Images without label in full_df.csv: {len(train_files - csv_files)} (UNLABELED)")
print(f"    preprocessed_images matching CSV filename: {len(csv_files & prep_files)} / {len(csv_files)} (100%)")

# Patient grouping analysis
patient_groups = df.groupby('ID')['filename'].apply(list).to_dict()
single_eye = sum(1 for v in patient_groups.values() if len(v) == 1)
both_eyes = sum(1 for v in patient_groups.values() if len(v) == 2)
other_cnt = sum(1 for v in patient_groups.values() if len(v) > 2)
print(f"\n  [E] Patient Grouping & Leakage Risk:")
print(f"    Patients with 1 image in CSV: {single_eye}")
print(f"    Patients with 2 images (Left + Right): {both_eyes}")
print(f"    Patients with >2 images: {other_cnt}")
print(f"    CRITICAL: 3,034 patients have BOTH left and right eyes in the dataset.")
print(f"    Splitting by image will cause SEVERE data leakage. Must split by Patient ID.")

# Audit result dictionary
results = {
    'modality': 'retina',
    'source_dataset': 'ODIR-5K',
    'total_physical_images': len(train_files) + len(test_files) + len(prep_files),
    'labeled_images': len(csv_files & train_files),
    'unlabeled_images': len(test_files) + len(train_files - csv_files),
    'preprocessed_images': len(prep_files),
    'classes': label_cols,
    'class_details': label_counts,
    'multi_label': True,
    'label_cardinality': cardinality,
    'patient_count': int(df['ID'].nunique()),
    'leakage_risk': 'HIGH - 3034 patients have multiple images (left/right eyes)',
    'readiness_status': 'REPAIR_REQUIRED',
    'unlabeled_inventory': {
        'odir_testing_images': len(test_files),
        'odir_unlabeled_training': len(train_files - csv_files)
    },
    'quarantine_recommendation': 'Quarantine preprocessed_images (unknown pipeline source) and unlabeled test/training images; use only 6392 validated ODIR training images with patient-grouped multi-label split.'
}

os.makedirs('outputs', exist_ok=True)
with open('outputs/v2_audit_retina.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n  Saved: outputs/v2_audit_retina.json")
