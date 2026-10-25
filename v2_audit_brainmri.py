"""Phase 1: Brain MRI Dataset Audit"""
import os, json
from pathlib import Path
from collections import defaultdict

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp'}
base = Path('datasets/brain_mri')

results = {'modality': 'brain_mri', 'splits': {}, 'duplicate_filenames': [], 'summary': {}}
all_names = defaultdict(list)
total = 0

for split in ['Training', 'Testing']:
    split_path = base / split
    results['splits'][split] = {}
    for cls_folder in sorted(split_path.iterdir()):
        if cls_folder.is_dir():
            img_files = [f for f in cls_folder.glob('*') if f.suffix.lower() in IMAGE_EXTS]
            non_img = [f for f in cls_folder.glob('*') if f.suffix.lower() not in IMAGE_EXTS and f.is_file()]
            exts = defaultdict(int)
            for f in img_files:
                exts[f.suffix.lower()] += 1
                all_names[f.name].append(str(f.relative_to(base)))
            results['splits'][split][cls_folder.name] = {
                'image_count': len(img_files),
                'non_image_count': len(non_img),
                'extensions': dict(exts)
            }
            total += len(img_files)

dup_names = {k: v for k, v in all_names.items() if len(v) > 1}
results['duplicate_filenames'] = list(dup_names.keys())[:20]
results['summary']['total_images'] = total
results['summary']['duplicate_filename_count'] = len(dup_names)
results['summary']['classes'] = list(results['splits']['Training'].keys())
results['summary']['label_source'] = 'folder_name'
results['summary']['patient_ids_available'] = False

print("=== BRAIN MRI AUDIT ===")
for split, classes in results['splits'].items():
    print(f"\n  {split}/")
    split_total = 0
    for cls, info in classes.items():
        print(f"    {cls}: {info['image_count']} images {info['extensions']}")
        split_total += info['image_count']
    print(f"    SPLIT TOTAL: {split_total}")

print(f"\n  GRAND TOTAL: {total}")
print(f"  Duplicate filenames across splits: {len(dup_names)}")
if dup_names:
    for name, paths in list(dup_names.items())[:5]:
        print(f"    DUPLICATE: {name}")
        for p in paths:
            print(f"      {p}")
print(f"\n  Classes confirmed: {results['summary']['classes']}")
print(f"  Label source: {results['summary']['label_source']}")
print(f"  Patient IDs: {results['summary']['patient_ids_available']}")

# Save result
os.makedirs('outputs', exist_ok=True)
with open('outputs/v2_audit_brainmri.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n  Saved: outputs/v2_audit_brainmri.json")
