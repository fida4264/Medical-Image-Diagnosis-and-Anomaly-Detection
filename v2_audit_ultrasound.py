"""Phase 1: Ultrasound Dataset Audit"""
import os, json
from pathlib import Path
from collections import defaultdict
import re

base = Path('datasets/Ultrasound/Dataset_BUSI_with_GT')
if not base.exists():
    base = Path('datasets/ultrasound/Dataset_BUSI_with_GT')

print("=== ULTRASOUND AUDIT ===")
print(f"  Base path: {base}")

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}

classes = ['benign', 'malignant', 'normal']
class_stats = {}
total_images = 0
total_masks = 0
mask_patterns = defaultdict(int)

# Check case issue
parent_dir = Path('datasets')
case_issues = []
for p in parent_dir.iterdir():
    if p.name.lower() == 'ultrasound':
        if p.name != 'ultrasound':
            case_issues.append(f"Directory is named '{p.name}' (capital U), but YAML configs reference 'ultrasound' (lowercase). On case-sensitive filesystems (Linux/Docker), this causes FileNotFoundError.")

print(f"\n  [A] Folder Naming & Case Audit:")
for issue in case_issues:
    print(f"    WARNING: {issue}")

print(f"\n  [B] Recursive Inventory per Class:")
detailed_mapping = {}

for cls in classes:
    cls_dir = base / cls
    if not cls_dir.exists():
        print(f"    Class dir {cls} not found!")
        continue

    all_files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]

    # Classify images vs masks
    # Masks have '_mask' in the name
    masks = []
    images = []

    for f in all_files:
        if '_mask' in f.name:
            masks.append(f)
            # Check pattern of mask (e.g. _mask.png, _mask_1.png)
            match = re.search(r'(_mask.*)\.png', f.name, re.IGNORECASE)
            if match:
                mask_patterns[match.group(1)] += 1
        else:
            images.append(f)

    # Pair images with masks
    paired_count = 0
    unpaired_images = []
    image_to_masks = defaultdict(list)

    for img in images:
        base_name = img.stem  # e.g., "benign (1)"
        # Find matching masks
        m_list = [m for m in masks if m.name.startswith(base_name + '_mask')]
        image_to_masks[img.name] = [m.name for m in m_list]
        if m_list:
            paired_count += 1
        else:
            unpaired_images.append(img.name)

    # Check for orphan masks (masks that don't match any image)
    orphan_masks = []
    for m in masks:
        # Extract image stem: everything before '_mask'
        idx = m.name.lower().find('_mask')
        parent_img_name = m.name[:idx] + m.suffix
        if not (cls_dir / parent_img_name).exists():
            orphan_masks.append(m.name)

    class_stats[cls] = {
        'total_files': len(all_files),
        'classification_images': len(images),
        'segmentation_masks': len(masks),
        'paired_images': paired_count,
        'unpaired_images': unpaired_images,
        'orphan_masks': orphan_masks
    }

    total_images += len(images)
    total_masks += len(masks)

    print(f"    {cls}:")
    print(f"      Total files: {len(all_files)}")
    print(f"      True classification images: {len(images)}")
    print(f"      Segmentation masks: {len(masks)}")
    print(f"      Images with matching mask(s): {paired_count} / {len(images)}")
    if unpaired_images:
        print(f"      Images without masks: {len(unpaired_images)} (e.g. {unpaired_images[:3]})")
    if orphan_masks:
        print(f"      Orphan masks: {len(orphan_masks)} (e.g. {orphan_masks[:3]})")

print(f"\n  [C] Modality Totals:")
print(f"    Total true classification images: {total_images}")
print(f"    Total segmentation mask files: {total_masks}")
print(f"    Total files combined: {total_images + total_masks}")
print(f"    Mask filename patterns found: {dict(mask_patterns)}")

print(f"\n  [D] Mask Contamination Risk Assessment:")
print(f"    CRITICAL: In V1 pipelines or naive folder loaders (e.g. torchvision ImageFolder),")
print(f"    all {total_images + total_masks} files would be loaded as classification images,")
print(f"    meaning masks would be treated as independent medical cases!")
print(f"    V2.0 must enforce strict mask filtering: only filenames WITHOUT '_mask' are classification images.")

audit_result = {
    'modality': 'ultrasound',
    'source_dataset': 'Dataset_BUSI_with_GT',
    'total_classification_images': total_images,
    'total_segmentation_masks': total_masks,
    'total_files': total_images + total_masks,
    'classes': classes,
    'class_details': class_stats,
    'mask_patterns': dict(mask_patterns),
    'case_sensitivity_issue': case_issues,
    'readiness_status': 'READY_WITH_CONDITIONS',
    'required_repairs': [
        'Filter out all files matching *_mask*.png from classification datasets',
        'Normalize folder path reference in configs to match actual filesystem case or add case-insensitive fallback',
        'Use stratified splitting due to small dataset size (780 total images)'
    ]
}

os.makedirs('outputs', exist_ok=True)
with open('outputs/v2_audit_ultrasound.json', 'w') as f:
    json.dump(audit_result, f, indent=2)

print("\n  Saved: outputs/v2_audit_ultrasound.json")
