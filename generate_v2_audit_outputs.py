"""
Generates comprehensive V2.0 Dataset Audit Outputs:
- outputs/v2_dataset_audit.json
- outputs/v2_dataset_inventory.csv
- outputs/v2_label_inventory.csv
"""
import os, json, csv
from pathlib import Path

os.makedirs('outputs', exist_ok=True)

# Load individual audit results
with open('outputs/v2_audit_brainmri.json') as f:
    brain_mri = json.load(f)
with open('outputs/v2_audit_brainct.json') as f:
    brain_ct = json.load(f)
with open('outputs/v2_audit_chestxray.json') as f:
    chest_xray = json.load(f)
with open('outputs/v2_audit_histopathology.json') as f:
    histopathology = json.load(f)
with open('outputs/v2_audit_retina.json') as f:
    retina = json.load(f)
with open('outputs/v2_audit_skin.json') as f:
    skin = json.load(f)
with open('outputs/v2_audit_ultrasound.json') as f:
    ultrasound = json.load(f)

# Build unified audit summary
v2_dataset_audit = {
    "version": "2.0.0",
    "phase": "PHASE 1 - DATASET INVENTORY & LABEL AUDIT",
    "timestamp": "2026-09-19",
    "summary_metrics": {
        "total_modalities": 7,
        "total_images_all_modalities": (
            brain_mri['summary']['total_images'] +
            brain_ct['total_images'] +
            chest_xray['covid_radiography']['total_images'] +
            histopathology['total_images'] +
            retina['total_physical_images'] +
            skin['total_physical_images'] +
            ultrasound['total_classification_images']
        ),
        "total_labeled_classification_images": (
            brain_mri['summary']['total_images'] +
            brain_ct['mapped_images'] +
            chest_xray['covid_radiography']['total_images'] +
            histopathology['total_images'] +
            retina['labeled_images'] +
            skin['matched_images'] +
            ultrasound['total_classification_images']
        ),
        "total_segmentation_masks": (
            chest_xray['covid_radiography']['total_masks'] +
            ultrasound['total_segmentation_masks']
        ),
        "total_unlabeled_images": (
            retina['unlabeled_images'] +
            retina['preprocessed_images']
        ),
        "hold_dataset_files": 18003
    },
    "modalities": {
        "brain_mri": {
            "dataset_name": "Brain Tumor MRI Dataset",
            "source_path": "datasets/brain_mri",
            "total_images": brain_mri['summary']['total_images'],
            "labeled_images": brain_mri['summary']['total_images'],
            "unlabeled_images": 0,
            "masks": 0,
            "classes": brain_mri['summary']['classes'],
            "class_counts": {
                "glioma": 1800,
                "meningioma": 1800,
                "notumor": 1800,
                "pituitary": 1800
            },
            "splits": {
                "training": 5600,
                "testing": 1600,
                "validation": 0
            },
            "multi_label": False,
            "patient_identifiers": False,
            "leakage_risk": "UNKNOWN (No patient IDs available; requires slice-level deduplication check)",
            "readiness_status": "READY_WITH_CONDITIONS",
            "conditions": [
                "Create reproducible stratified validation split from Training directory",
                "Evaluate model against full 1,600 test images instead of tiny 32-sample subset"
            ]
        },
        "brain_ct": {
            "dataset_name": "Head CT Hemorrhage Dataset",
            "source_path": "datasets/brain_ct",
            "total_images": brain_ct['total_images'],
            "labeled_images": brain_ct['mapped_images'],
            "unlabeled_images": 0,
            "masks": 0,
            "classes": ["hemorrhage", "normal"],
            "class_counts": brain_ct['label_counts'],
            "splits": {"training": 0, "testing": 0, "validation": 0},
            "multi_label": False,
            "patient_identifiers": False,
            "leakage_risk": "LOW (Unique numeric scan IDs 0-199)",
            "readiness_status": "REPAIR_REQUIRED",
            "conditions": [
                "CRITICAL: Retrain model from scratch. V1 model was trained as 1-class 'head_ct' ignoring binary hemorrhage labels",
                "Implement stratified k-fold cross-validation due to small sample size (200 images)"
            ]
        },
        "chest_xray": {
            "dataset_name": "COVID-19 Radiography Dataset",
            "source_path": "datasets/chest_xray/COVID-19_Radiography_Dataset",
            "total_images": chest_xray['covid_radiography']['total_images'],
            "labeled_images": chest_xray['covid_radiography']['total_images'],
            "unlabeled_images": 0,
            "masks": chest_xray['covid_radiography']['total_masks'],
            "classes": ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"],
            "class_counts": {k: v['image_count'] for k, v in chest_xray['covid_radiography']['class_stats'].items()},
            "splits": {"training": 0, "testing": 0, "validation": 0},
            "multi_label": False,
            "patient_identifiers": False,
            "leakage_risk": "LOW (Standard benchmark dataset)",
            "readiness_status": "READY_WITH_CONDITIONS",
            "conditions": [
                "Exclude all 21,165 mask files from classification training",
                "Remove spurious 'unknown' class from V1 model/config",
                "VinBigData (18,000 detection images) remains HOLD"
            ]
        },
        "histopathology": {
            "dataset_name": "LC25000 Lung and Colon Histopathology",
            "source_path": "datasets/histopathology/lung_colon_image_set",
            "total_images": histopathology['total_images'],
            "labeled_images": histopathology['total_images'],
            "unlabeled_images": 0,
            "masks": 0,
            "classes": list(histopathology['class_stats'].keys()),
            "class_counts": {k: v['count'] for k, v in histopathology['class_stats'].items()},
            "splits": {"training": 0, "testing": 0, "validation": 0},
            "multi_label": False,
            "patient_identifiers": False,
            "leakage_risk": "MEDIUM (Known tile-level near-duplicates in LC25000 benchmark)",
            "readiness_status": "READY",
            "conditions": [
                "Enforce random seed preservation across splits to prevent train/test tile leakage"
            ]
        },
        "retina": {
            "dataset_name": "ODIR-5K (Ocular Disease Intelligent Recognition)",
            "source_path": "datasets/retina",
            "total_images": retina['total_physical_images'],
            "labeled_images": retina['labeled_images'],
            "unlabeled_images": retina['unlabeled_images'],
            "preprocessed_images": retina['preprocessed_images'],
            "masks": 0,
            "classes": retina['classes'],
            "class_counts": {k: v['count'] for k, v in retina['class_details'].items()},
            "splits": {"training": 7000, "testing": 1000},
            "multi_label": True,
            "patient_identifiers": True,
            "leakage_risk": "HIGH (3,034 patients have both Left and Right eye images)",
            "readiness_status": "REPAIR_REQUIRED",
            "conditions": [
                "Implement patient-level grouping using Patient 'ID' to prevent left/right eye leakage across splits",
                "Reformulate training pipeline as true multi-label with BCEWithLogitsLoss",
                "Quarantine 6,392 preprocessed_images (UNKNOWN_SOURCE) and 1,608 unlabeled test/train images"
            ]
        },
        "skin": {
            "dataset_name": "HAM10000 Skin Lesions",
            "source_path": "datasets/skin",
            "total_images": skin['total_physical_images'],
            "labeled_images": skin['matched_images'],
            "unlabeled_images": 0,
            "masks": 0,
            "classes": skin['classes'],
            "class_counts": {k: v['count'] for k, v in skin['class_details'].items()},
            "splits": {"training": 0, "testing": 0, "validation": 0},
            "multi_label": False,
            "patient_identifiers": True,
            "leakage_risk": "HIGH (7,470 lesions across 10,015 images; 2,545 duplicate lesion images)",
            "readiness_status": "REPAIR_REQUIRED",
            "conditions": [
                "CRITICAL: Perform metadata join between HAM10000_metadata.csv and image paths (was omitted in V1)",
                "Implement lesion-level grouping using 'lesion_id' to prevent duplicate lesion leakage across splits",
                "Address extreme 58.3:1 class imbalance (nv: 6,705 vs df: 115) using class weighting or focal loss"
            ]
        },
        "ultrasound": {
            "dataset_name": "Dataset_BUSI_with_GT (Breast Ultrasound Images)",
            "source_path": "datasets/Ultrasound/Dataset_BUSI_with_GT",
            "total_images": ultrasound['total_classification_images'],
            "labeled_images": ultrasound['total_classification_images'],
            "unlabeled_images": 0,
            "masks": ultrasound['total_segmentation_masks'],
            "classes": ultrasound['classes'],
            "class_counts": {k: v['classification_images'] for k, v in ultrasound['class_details'].items()},
            "splits": {"training": 0, "testing": 0, "validation": 0},
            "multi_label": False,
            "patient_identifiers": False,
            "leakage_risk": "LOW (Standard BUSI benchmark)",
            "readiness_status": "READY_WITH_CONDITIONS",
            "conditions": [
                "Exclude all 798 segmentation mask files from classification dataset",
                "Fix folder case sensitivity issue ('Ultrasound' vs 'ultrasound') in configs/loaders",
                "Use stratified k-fold cross validation due to small dataset size (780 images)"
            ]
        }
    }
}

with open('outputs/v2_dataset_audit.json', 'w') as f:
    json.dump(v2_dataset_audit, f, indent=2)
print("Saved outputs/v2_dataset_audit.json")

# Build outputs/v2_dataset_inventory.csv
inv_rows = [
    {
        "Modality": "Brain MRI",
        "Dataset Name": "Brain Tumor MRI Dataset",
        "Total Files": 7200,
        "Classification Images": 7200,
        "Masks": 0,
        "Labeled Images": 7200,
        "Unlabeled Images": 0,
        "Classes Count": 4,
        "Classes": "glioma, meningioma, notumor, pituitary",
        "Multi-label": "No",
        "Pre-existing Splits": "Yes (Train: 5600, Test: 1600)",
        "Patient IDs": "No",
        "Leakage Risk": "Unknown",
        "Readiness Status": "READY_WITH_CONDITIONS"
    },
    {
        "Modality": "Brain CT",
        "Dataset Name": "Head CT Hemorrhage Dataset",
        "Total Files": 201,
        "Classification Images": 200,
        "Masks": 0,
        "Labeled Images": 200,
        "Unlabeled Images": 0,
        "Classes Count": 2,
        "Classes": "hemorrhage, normal",
        "Multi-label": "No",
        "Pre-existing Splits": "No",
        "Patient IDs": "No",
        "Leakage Risk": "Low",
        "Readiness Status": "REPAIR_REQUIRED"
    },
    {
        "Modality": "Chest X-ray",
        "Dataset Name": "COVID-19 Radiography Dataset",
        "Total Files": 42330,
        "Classification Images": 21165,
        "Masks": 21165,
        "Labeled Images": 21165,
        "Unlabeled Images": 0,
        "Classes Count": 4,
        "Classes": "COVID, Lung_Opacity, Normal, Viral Pneumonia",
        "Multi-label": "No",
        "Pre-existing Splits": "No",
        "Patient IDs": "No",
        "Leakage Risk": "Low",
        "Readiness Status": "READY_WITH_CONDITIONS"
    },
    {
        "Modality": "Histopathology",
        "Dataset Name": "LC25000 Lung & Colon",
        "Total Files": 25000,
        "Classification Images": 25000,
        "Masks": 0,
        "Labeled Images": 25000,
        "Unlabeled Images": 0,
        "Classes Count": 5,
        "Classes": "colon_aca, colon_n, lung_aca, lung_n, lung_scc",
        "Multi-label": "No",
        "Pre-existing Splits": "No",
        "Patient IDs": "No",
        "Leakage Risk": "Medium (Tile Near-Duplicates)",
        "Readiness Status": "READY"
    },
    {
        "Modality": "Retina",
        "Dataset Name": "ODIR-5K",
        "Total Files": 14392,
        "Classification Images": 7000,
        "Masks": 0,
        "Labeled Images": 6392,
        "Unlabeled Images": 1608,
        "Classes Count": 8,
        "Classes": "N, D, G, C, A, H, M, O",
        "Multi-label": "Yes",
        "Pre-existing Splits": "Yes (Train: 7000, Test: 1000)",
        "Patient IDs": "Yes (3,358 unique)",
        "Leakage Risk": "High (Left/Right eyes)",
        "Readiness Status": "REPAIR_REQUIRED"
    },
    {
        "Modality": "Skin Lesion",
        "Dataset Name": "HAM10000",
        "Total Files": 10015,
        "Classification Images": 10015,
        "Masks": 0,
        "Labeled Images": 10015,
        "Unlabeled Images": 0,
        "Classes Count": 7,
        "Classes": "nv, mel, bkl, bcc, akiec, vasc, df",
        "Multi-label": "No",
        "Pre-existing Splits": "No",
        "Patient IDs": "Yes (7,470 lesions)",
        "Leakage Risk": "High (Multiple lesion images)",
        "Readiness Status": "REPAIR_REQUIRED"
    },
    {
        "Modality": "Ultrasound",
        "Dataset Name": "Dataset_BUSI_with_GT",
        "Total Files": 1578,
        "Classification Images": 780,
        "Masks": 798,
        "Labeled Images": 780,
        "Unlabeled Images": 0,
        "Classes Count": 3,
        "Classes": "benign, malignant, normal",
        "Multi-label": "No",
        "Pre-existing Splits": "No",
        "Patient IDs": "No",
        "Leakage Risk": "Low",
        "Readiness Status": "READY_WITH_CONDITIONS"
    }
]

with open('outputs/v2_dataset_inventory.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(inv_rows[0].keys()))
    writer.writeheader()
    writer.writerows(inv_rows)
print("Saved outputs/v2_dataset_inventory.csv")

# Build outputs/v2_label_inventory.csv
label_rows = [
    # Brain MRI
    {"Modality": "Brain MRI", "Raw Label": "glioma", "Standardized Class": "Glioma", "Sample Count": 1800, "Percentage": 25.0, "Label Status": "VALIDATED", "Verification Source": "Folder structure", "Notes": "1400 train + 400 test"},
    {"Modality": "Brain MRI", "Raw Label": "meningioma", "Standardized Class": "Meningioma", "Sample Count": 1800, "Percentage": 25.0, "Label Status": "VALIDATED", "Verification Source": "Folder structure", "Notes": "1400 train + 400 test"},
    {"Modality": "Brain MRI", "Raw Label": "notumor", "Standardized Class": "No Tumor / Normal", "Sample Count": 1800, "Percentage": 25.0, "Label Status": "VALIDATED", "Verification Source": "Folder structure", "Notes": "1400 train + 400 test"},
    {"Modality": "Brain MRI", "Raw Label": "pituitary", "Standardized Class": "Pituitary Tumor", "Sample Count": 1800, "Percentage": 25.0, "Label Status": "VALIDATED", "Verification Source": "Folder structure", "Notes": "1400 train + 400 test"},

    # Brain CT
    {"Modality": "Brain CT", "Raw Label": "1", "Standardized Class": "Hemorrhage", "Sample Count": 100, "Percentage": 50.0, "Label Status": "VALIDATED", "Verification Source": "labels.csv (hemorrhage=1)", "Notes": "Numeric ID mapping 000-099"},
    {"Modality": "Brain CT", "Raw Label": "0", "Standardized Class": "Normal", "Sample Count": 100, "Percentage": 50.0, "Label Status": "VALIDATED", "Verification Source": "labels.csv (hemorrhage=0)", "Notes": "Numeric ID mapping 100-199"},

    # Chest X-ray
    {"Modality": "Chest X-ray", "Raw Label": "COVID", "Standardized Class": "COVID-19", "Sample Count": 3616, "Percentage": 17.08, "Label Status": "VALIDATED", "Verification Source": "Folder structure + XLSX", "Notes": "Paired with 3,616 masks"},
    {"Modality": "Chest X-ray", "Raw Label": "Lung_Opacity", "Standardized Class": "Lung Opacity", "Sample Count": 6012, "Percentage": 28.41, "Label Status": "VALIDATED", "Verification Source": "Folder structure + XLSX", "Notes": "Paired with 6,012 masks"},
    {"Modality": "Chest X-ray", "Raw Label": "Normal", "Standardized Class": "Normal", "Sample Count": 10192, "Percentage": 48.15, "Label Status": "VALIDATED", "Verification Source": "Folder structure + XLSX", "Notes": "Paired with 10,192 masks"},
    {"Modality": "Chest X-ray", "Raw Label": "Viral Pneumonia", "Standardized Class": "Viral Pneumonia", "Sample Count": 1345, "Percentage": 6.35, "Label Status": "VALIDATED", "Verification Source": "Folder structure + XLSX", "Notes": "Paired with 1,345 masks"},
    {"Modality": "Chest X-ray", "Raw Label": "unknown", "Standardized Class": "INVALID_CLASS", "Sample Count": 0, "Percentage": 0.0, "Label Status": "INVALID_LABEL", "Verification Source": "None (V1 artifact bug)", "Notes": "Spurious class in V1 model, 0 samples in source"},

    # Histopathology
    {"Modality": "Histopathology", "Raw Label": "colon_aca", "Standardized Class": "Colon Adenocarcinoma", "Sample Count": 5000, "Percentage": 20.0, "Label Status": "VALIDATED", "Verification Source": "Nested folder structure", "Notes": "768x768 RGB"},
    {"Modality": "Histopathology", "Raw Label": "colon_n", "Standardized Class": "Colon Benign/Normal", "Sample Count": 5000, "Percentage": 20.0, "Label Status": "VALIDATED", "Verification Source": "Nested folder structure", "Notes": "768x768 RGB"},
    {"Modality": "Histopathology", "Raw Label": "lung_aca", "Standardized Class": "Lung Adenocarcinoma", "Sample Count": 5000, "Percentage": 20.0, "Label Status": "VALIDATED", "Verification Source": "Nested folder structure", "Notes": "768x768 RGB"},
    {"Modality": "Histopathology", "Raw Label": "lung_n", "Standardized Class": "Lung Benign/Normal", "Sample Count": 5000, "Percentage": 20.0, "Label Status": "VALIDATED", "Verification Source": "Nested folder structure", "Notes": "768x768 RGB"},
    {"Modality": "Histopathology", "Raw Label": "lung_scc", "Standardized Class": "Lung Squamous Cell Carcinoma", "Sample Count": 5000, "Percentage": 20.0, "Label Status": "VALIDATED", "Verification Source": "Nested folder structure", "Notes": "768x768 RGB"},

    # Retina
    {"Modality": "Retina", "Raw Label": "N", "Standardized Class": "Normal", "Sample Count": 2101, "Percentage": 32.87, "Label Status": "VALIDATED", "Verification Source": "full_df.csv (N=1)", "Notes": "Multi-label target"},
    {"Modality": "Retina", "Raw Label": "D", "Standardized Class": "Diabetic Retinopathy", "Sample Count": 2123, "Percentage": 33.21, "Label Status": "VALIDATED", "Verification Source": "full_df.csv (D=1)", "Notes": "Multi-label target"},
    {"Modality": "Retina", "Raw Label": "G", "Standardized Class": "Glaucoma", "Sample Count": 397, "Percentage": 6.21, "Label Status": "VALIDATED", "Verification Source": "full_df.csv (G=1)", "Notes": "Multi-label target"},
    {"Modality": "Retina", "Raw Label": "C", "Standardized Class": "Cataract", "Sample Count": 402, "Percentage": 6.29, "Label Status": "VALIDATED", "Verification Source": "full_df.csv (C=1)", "Notes": "Multi-label target"},
    {"Modality": "Retina", "Raw Label": "A", "Standardized Class": "Age-related Macular Degeneration", "Sample Count": 319, "Percentage": 4.99, "Label Status": "VALIDATED", "Verification Source": "full_df.csv (A=1)", "Notes": "Multi-label target"},
    {"Modality": "Retina", "Raw Label": "H", "Standardized Class": "Hypertension", "Sample Count": 203, "Percentage": 3.18, "Label Status": "VALIDATED", "Verification Source": "full_df.csv (H=1)", "Notes": "Multi-label target"},
    {"Modality": "Retina", "Raw Label": "M", "Standardized Class": "Pathological Myopia", "Sample Count": 306, "Percentage": 4.79, "Label Status": "VALIDATED", "Verification Source": "full_df.csv (M=1)", "Notes": "Multi-label target"},
    {"Modality": "Retina", "Raw Label": "O", "Standardized Class": "Other Disease/Abnormality", "Sample Count": 1588, "Percentage": 24.84, "Label Status": "VALIDATED", "Verification Source": "full_df.csv (O=1)", "Notes": "Multi-label target"},
    {"Modality": "Retina", "Raw Label": "preprocessed_images", "Standardized Class": "UNKNOWN", "Sample Count": 6392, "Percentage": 0.0, "Label Status": "UNKNOWN_SOURCE", "Verification Source": "Unknown pipeline", "Notes": "Quarantine; do not use as supervised training"},
    {"Modality": "Retina", "Raw Label": "ODIR-5K Testing Images", "Standardized Class": "UNLABELED", "Sample Count": 1000, "Percentage": 0.0, "Label Status": "MISSING_LABEL", "Verification Source": "Competition test set", "Notes": "Quarantine; no ground truth"},
    {"Modality": "Retina", "Raw Label": "ODIR-5K Unlabeled Train", "Standardized Class": "UNLABELED", "Sample Count": 608, "Percentage": 0.0, "Label Status": "MISSING_LABEL", "Verification Source": "No record in full_df.csv", "Notes": "Quarantine; 608 files in train dir have no CSV entry"},

    # Skin Lesion
    {"Modality": "Skin Lesion", "Raw Label": "nv", "Standardized Class": "Melanocytic Nevi", "Sample Count": 6705, "Percentage": 66.95, "Label Status": "VALIDATED", "Verification Source": "HAM10000_metadata.csv", "Notes": "66.95% of dataset"},
    {"Modality": "Skin Lesion", "Raw Label": "mel", "Standardized Class": "Melanoma", "Sample Count": 1113, "Percentage": 11.11, "Label Status": "VALIDATED", "Verification Source": "HAM10000_metadata.csv", "Notes": "11.11% of dataset"},
    {"Modality": "Skin Lesion", "Raw Label": "bkl", "Standardized Class": "Benign Keratosis", "Sample Count": 1099, "Percentage": 10.97, "Label Status": "VALIDATED", "Verification Source": "HAM10000_metadata.csv", "Notes": "10.97% of dataset"},
    {"Modality": "Skin Lesion", "Raw Label": "bcc", "Standardized Class": "Basal Cell Carcinoma", "Sample Count": 514, "Percentage": 5.13, "Label Status": "VALIDATED", "Verification Source": "HAM10000_metadata.csv", "Notes": "5.13% of dataset"},
    {"Modality": "Skin Lesion", "Raw Label": "akiec", "Standardized Class": "Actinic Keratoses", "Sample Count": 327, "Percentage": 3.27, "Label Status": "VALIDATED", "Verification Source": "HAM10000_metadata.csv", "Notes": "3.27% of dataset"},
    {"Modality": "Skin Lesion", "Raw Label": "vasc", "Standardized Class": "Vascular Lesions", "Sample Count": 142, "Percentage": 1.42, "Label Status": "VALIDATED", "Verification Source": "HAM10000_metadata.csv", "Notes": "1.42% of dataset"},
    {"Modality": "Skin Lesion", "Raw Label": "df", "Standardized Class": "Dermatofibroma", "Sample Count": 115, "Percentage": 1.15, "Label Status": "VALIDATED", "Verification Source": "HAM10000_metadata.csv", "Notes": "1.15% of dataset"},

    # Ultrasound
    {"Modality": "Ultrasound", "Raw Label": "benign", "Standardized Class": "Benign", "Sample Count": 437, "Percentage": 56.03, "Label Status": "VALIDATED", "Verification Source": "Folder structure", "Notes": "454 paired masks excluded"},
    {"Modality": "Ultrasound", "Raw Label": "malignant", "Standardized Class": "Malignant", "Sample Count": 210, "Percentage": 26.92, "Label Status": "VALIDATED", "Verification Source": "Folder structure", "Notes": "211 paired masks excluded"},
    {"Modality": "Ultrasound", "Raw Label": "normal", "Standardized Class": "Normal", "Sample Count": 133, "Percentage": 17.05, "Label Status": "VALIDATED", "Verification Source": "Folder structure", "Notes": "133 paired masks excluded"},
    {"Modality": "Ultrasound", "Raw Label": "*_mask*.png", "Standardized Class": "SEGMENTATION_MASK", "Sample Count": 798, "Percentage": 0.0, "Label Status": "MASK", "Verification Source": "BUSI GT annotations", "Notes": "Must not be used as classification images"}
]

with open('outputs/v2_label_inventory.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(label_rows[0].keys()))
    writer.writeheader()
    writer.writerows(label_rows)
print("Saved outputs/v2_label_inventory.csv")
print("\nAll V2.0 dataset audit outputs generated successfully.")
