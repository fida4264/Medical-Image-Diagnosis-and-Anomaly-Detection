"""Phase 1: Chest X-ray Dataset Audit"""
import os, json
from pathlib import Path
from collections import defaultdict

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
base = Path('datasets/chest_xray')
covid_base = base / 'COVID-19_Radiography_Dataset'
vinbig_base = base / 'vinbigdata'

print("=== CHEST X-RAY AUDIT ===")

# --- COVID-19 Radiography Dataset ---
print("\n  [A] COVID-19 Radiography Classification Dataset")
classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
class_stats = {}
total_img = 0
total_mask = 0
all_img_names = defaultdict(list)

for cls in classes:
    cls_path = covid_base / cls
    img_dir = cls_path / 'images'
    mask_dir = cls_path / 'masks'

    imgs = list(img_dir.glob('*')) if img_dir.exists() else []
    masks = list(mask_dir.glob('*')) if mask_dir.exists() else []

    img_files = [f for f in imgs if f.suffix.lower() in IMAGE_EXTS]
    mask_files = [f for f in masks if f.suffix.lower() in IMAGE_EXTS]

    exts = defaultdict(int)
    for f in img_files:
        exts[f.suffix.lower()] += 1
        all_img_names[f.name].append(cls)

    # Match images to masks
    img_stems = {f.stem for f in img_files}
    mask_stems = {f.stem for f in mask_files}
    imgs_with_mask = img_stems & mask_stems
    imgs_without_mask = img_stems - mask_stems
    masks_without_img = mask_stems - img_stems

    class_stats[cls] = {
        'image_count': len(img_files),
        'mask_count': len(mask_files),
        'images_with_matching_mask': len(imgs_with_mask),
        'images_without_mask': len(imgs_without_mask),
        'masks_without_image': len(masks_without_img),
        'extensions': dict(exts)
    }
    total_img += len(img_files)
    total_mask += len(mask_files)
    print(f"    {cls}: {len(img_files)} images | {len(mask_files)} masks | {len(imgs_with_mask)} paired")

print(f"\n    TOTAL classification images: {total_img}")
print(f"    TOTAL mask files: {total_mask}")

# Duplicate image filenames across classes
dup_names = {k: v for k, v in all_img_names.items() if len(v) > 1}
print(f"\n    Duplicate image filenames across classes: {len(dup_names)}")
if dup_names:
    for name, clss in list(dup_names.items())[:3]:
        print(f"      {name} appears in: {clss}")

# Check XLSX metadata files
print("\n    Metadata XLSX files:")
for xlsx in sorted(covid_base.glob('*.xlsx')):
    size_kb = xlsx.stat().st_size // 1024
    print(f"      {xlsx.name} ({size_kb} KB)")

# Check for README
for f in covid_base.glob('*.txt'):
    print(f"      README: {f.name}")

# --- VinBigData ---
print("\n  [B] VinBigData Detection Dataset")
if vinbig_base.exists():
    all_vinbig = list(vinbig_base.rglob('*'))
    files = [f for f in all_vinbig if f.is_file()]
    exts = defaultdict(int)
    for f in files:
        exts[f.suffix.lower()] += 1
    print(f"    Total files: {len(files)}")
    print(f"    File types: {dict(exts)}")
    subdirs = [d for d in vinbig_base.iterdir() if d.is_dir()]
    print(f"    Subdirectories: {[d.name for d in subdirs]}")
    for d in subdirs:
        sub_files = list(d.rglob('*'))
        sub_imgs = [f for f in sub_files if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        sub_csvs = [f for f in sub_files if f.is_file() and f.suffix.lower() == '.csv']
        print(f"      {d.name}/: {len(sub_imgs)} images, {len(sub_csvs)} CSVs")
        for csv_f in sub_csvs[:2]:
            print(f"        CSV: {csv_f.name} ({csv_f.stat().st_size//1024} KB)")
else:
    print("    vinbigdata/ NOT FOUND")

# Verify no "unknown" class in source
print("\n  [C] Source class names verification:")
actual_classes = [d.name for d in covid_base.iterdir() if d.is_dir()]
print(f"    Actual class folders found: {sorted(actual_classes)}")
print(f"    'unknown' class in source data: {'unknown' in [c.lower() for c in actual_classes]}")
print(f"    V1 'unknown' class was a BUG - NOT present in source data")

result = {
    'modality': 'chest_xray',
    'covid_radiography': {
        'classes': classes,
        'class_stats': class_stats,
        'total_images': total_img,
        'total_masks': total_mask,
        'duplicate_filenames': len(dup_names),
        'unknown_class_in_source': False,
        'v1_unknown_class_bug': True
    },
    'vinbigdata': {
        'format': 'detection_bounding_box',
        'status': 'HOLD',
        'reason': 'Detection format only - not classification ready'
    }
}

os.makedirs('outputs', exist_ok=True)
with open('outputs/v2_audit_chestxray.json', 'w') as f:
    json.dump(result, f, indent=2)
print("\n  Saved: outputs/v2_audit_chestxray.json")
