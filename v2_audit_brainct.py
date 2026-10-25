"""Phase 1: Brain CT Dataset Audit"""
import pandas as pd, json, os
from pathlib import Path

base = Path('datasets/brain_ct')
csv_path = base / 'labels.csv'
img_dir = base / 'head_ct'

df = pd.read_csv(csv_path)
# Strip whitespace from column names
df.columns = [c.strip() for c in df.columns]

print("=== BRAIN CT AUDIT ===")
print(f"\n  labels.csv:")
print(f"    Rows: {len(df)}")
print(f"    Columns: {list(df.columns)}")
print(f"    Unique label values: {sorted(df['hemorrhage'].unique().tolist())}")
print(f"    hemorrhage=1 (positive): {(df['hemorrhage']==1).sum()}")
print(f"    hemorrhage=0 (normal): {(df['hemorrhage']==0).sum()}")
print(f"    Missing in id: {df['id'].isnull().sum()}")
print(f"    Missing in hemorrhage: {df['hemorrhage'].isnull().sum()}")
print(f"    ID range: {int(df['id'].min())} to {int(df['id'].max())}")

img_files = sorted(img_dir.glob('*.png'))
print(f"\n  Image files in head_ct/: {len(img_files)}")
other_files = [f for f in img_dir.glob('*') if f.suffix.lower() != '.png']
print(f"  Non-PNG files in head_ct/: {len(other_files)}")

img_ids = {}
bad_names = []
for f in img_files:
    try:
        img_ids[int(f.stem)] = f.name
    except ValueError:
        bad_names.append(f.name)

csv_ids = set(df['id'].tolist())
img_id_set = set(img_ids.keys())
in_csv_not_img = csv_ids - img_id_set
in_img_not_csv = img_id_set - csv_ids

print(f"\n  Cross-reference:")
print(f"    IDs in CSV but no matching image: {len(in_csv_not_img)}")
print(f"    Images with no CSV label: {len(in_img_not_csv)}")
print(f"    Successfully mapped: {len(img_id_set & csv_ids)}")
print(f"    Non-numeric filenames: {bad_names}")

print("\n  Sample image->label mapping (first 10):")
mapping = []
for f in sorted(img_files)[:10]:
    fid = int(f.stem)
    rows = df[df['id'] == fid]
    if len(rows) > 0:
        hval = int(rows['hemorrhage'].values[0])
        std = 'hemorrhage' if hval == 1 else 'normal'
        lstat = 'VALIDATED'
    else:
        hval = 'NO_CSV_ENTRY'
        std = 'MISSING_LABEL'
        lstat = 'MISSING_LABEL'
    print(f"    {f.name} -> id={fid} -> hemorrhage={hval} -> class={std} -> status={lstat}")
    mapping.append({'filename': f.name, 'id': fid, 'hemorrhage': hval, 'class': std, 'label_status': lstat})

print("\n  Full mapping (all 200 images):")
full_mapping = []
for f in sorted(img_files):
    fid = int(f.stem)
    rows = df[df['id'] == fid]
    if len(rows) > 0:
        hval = int(rows['hemorrhage'].values[0])
        std = 'hemorrhage' if hval == 1 else 'normal'
        lstat = 'VALIDATED'
    else:
        hval = None
        std = None
        lstat = 'MISSING_LABEL'
    full_mapping.append({'filename': f.name, 'id': fid, 'hemorrhage': hval, 'standardized_class': std, 'label_status': lstat})

status_counts = {}
for item in full_mapping:
    s = item['label_status']
    status_counts[s] = status_counts.get(s, 0) + 1
print(f"    Label status counts: {status_counts}")

class_counts = {}
for item in full_mapping:
    c = item['standardized_class']
    if c:
        class_counts[c] = class_counts.get(c, 0) + 1
print(f"    Class counts: {class_counts}")

result = {
    'modality': 'brain_ct',
    'csv_rows': len(df),
    'csv_columns': list(df.columns),
    'label_counts': {'hemorrhage': int((df['hemorrhage']==1).sum()), 'normal': int((df['hemorrhage']==0).sum())},
    'total_images': len(img_files),
    'mapped_images': len(img_id_set & csv_ids),
    'unmapped_images': len(in_img_not_csv),
    'labels_without_image': list(in_csv_not_img),
    'class_counts': class_counts,
    'label_status_counts': status_counts,
    'full_mapping': full_mapping
}

os.makedirs('outputs', exist_ok=True)
with open('outputs/v2_audit_brainct.json', 'w') as f:
    json.dump(result, f, indent=2)
print("\n  Saved: outputs/v2_audit_brainct.json")
