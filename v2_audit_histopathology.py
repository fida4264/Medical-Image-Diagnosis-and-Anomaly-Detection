"""Phase 1: Histopathology Dataset Audit"""
import os, json
from pathlib import Path
from collections import defaultdict

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
base = Path('datasets/histopathology/lung_colon_image_set')

print("=== HISTOPATHOLOGY AUDIT ===")
print(f"\n  Base path: {base}")

# Recursive directory walk
def walk_dirs(path, depth=0):
    items = []
    try:
        for item in sorted(path.iterdir()):
            if item.is_dir():
                print(f"  {'  '*depth}[DIR] {item.name}/")
                items.extend(walk_dirs(item, depth+1))
            elif item.is_file():
                items.append(item)
    except Exception as e:
        print(f"  ERROR: {e}")
    return items

print("\n  Directory tree:")
all_files = walk_dirs(base)

# Group by class folder
IMAGE_EXTS_check = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
class_stats = {}
all_filenames = defaultdict(list)

for f in all_files:
    if f.is_file() and f.suffix.lower() in IMAGE_EXTS_check:
        cls_name = f.parent.name
        if cls_name not in class_stats:
            class_stats[cls_name] = {'count': 0, 'extensions': defaultdict(int), 'parent_dir': f.parent.parent.name}
        class_stats[cls_name]['count'] += 1
        class_stats[cls_name]['extensions'][f.suffix.lower()] += 1
        all_filenames[f.name].append(cls_name)

print(f"\n  Classes found: {sorted(class_stats.keys())}")
print(f"\n  Per-class image counts:")
total = 0
for cls in sorted(class_stats.keys()):
    info = class_stats[cls]
    exts = dict(info['extensions'])
    print(f"    {info['parent_dir']}/{cls}: {info['count']} images | exts: {exts}")
    total += info['count']
print(f"\n  TOTAL images: {total}")

# Duplicate filenames
dup_names = {k: v for k, v in all_filenames.items() if len(v) > 1}
print(f"\n  Duplicate filenames across classes: {len(dup_names)}")
if dup_names:
    for name, clss in list(dup_names.items())[:3]:
        print(f"    DUPLICATE: {name} in {clss}")

# Non-image files
non_img = [f for f in all_files if f.suffix.lower() not in IMAGE_EXTS_check]
print(f"\n  Non-image files found: {len(non_img)}")
for f in non_img[:10]:
    print(f"    {f}")

# Check sample image dimensions
try:
    from PIL import Image
    samples = {}
    for cls in sorted(class_stats.keys()):
        for f in all_files:
            if f.parent.name == cls and f.suffix.lower() in IMAGE_EXTS_check:
                try:
                    with Image.open(f) as img:
                        samples[cls] = {'size': img.size, 'mode': img.mode}
                    break
                except:
                    pass
    print(f"\n  Sample image dimensions per class:")
    for cls, info in samples.items():
        print(f"    {cls}: {info['size']} mode={info['mode']}")
except ImportError:
    print("  PIL not available for dimension check")

# Class mapping
class_name_mapping = {
    'colon_aca': 'Colon Adenocarcinoma',
    'colon_n': 'Colon Benign/Normal',
    'lung_aca': 'Lung Adenocarcinoma',
    'lung_n': 'Lung Benign/Normal',
    'lung_scc': 'Lung Squamous Cell Carcinoma'
}
print(f"\n  Standardized class name mapping:")
for raw, std in class_name_mapping.items():
    if raw in class_stats:
        print(f"    {raw} -> {std} ({class_stats[raw]['count']} images)")
    else:
        print(f"    {raw} -> NOT FOUND IN DATASET")

result = {
    'modality': 'histopathology',
    'dataset': 'LC25000',
    'class_stats': {k: {'count': v['count'], 'parent': v['parent_dir'], 'extensions': dict(v['extensions'])} for k, v in class_stats.items()},
    'total_images': total,
    'duplicate_filenames': len(dup_names),
    'non_image_files': len(non_img),
    'class_mapping': class_name_mapping
}

os.makedirs('outputs', exist_ok=True)
with open('outputs/v2_audit_histopathology.json', 'w') as f:
    json.dump(result, f, indent=2)
print("\n  Saved: outputs/v2_audit_histopathology.json")
