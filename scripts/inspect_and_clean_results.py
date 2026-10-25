import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "outputs" / "experiments"
CSV_PATH = ROOT / "outputs" / "v2_baseline_results.csv"

def inspect_all():
    print("=== INSPECTING ALL EXPERIMENT RESULTS ===")
    results = []

    # 1. K-Fold modalities
    for mod in ["brain_ct", "ultrasound"]:
        mod_dir = EXP_DIR / mod
        f_accs, f_f1s, f_times = [], [], []
        print(f"\n--- {mod.upper()} (5-Fold CV) ---")
        for f in range(5):
            fold_file = mod_dir / f"fold_{f}" / "experiment_results.json"
            if fold_file.exists():
                with open(fold_file, "r") as fp:
                    data = json.load(fp)
                bvm = data.get("best_val_metrics", {})
                time_s = data.get("training_time_seconds", 0)
                f_accs.append(bvm.get("accuracy", 0))
                f_f1s.append(bvm.get("macro_f1", 0))
                f_times.append(time_s)
                print(f"  Fold {f}: Acc={bvm.get('accuracy'):.4f}, Macro-F1={bvm.get('macro_f1'):.4f}, Time={time_s:.1f}s")
                results.append({
                    "modality": mod,
                    "model": "resnet18",
                    "dataset_version": "2.0.0",
                    "seed": 42,
                    "split_strategy": "stratified_5fold_cv",
                    "fold": str(f),
                    "accuracy": f"{bvm.get('accuracy'):.4f}",
                    "macro_f1": f"{bvm.get('macro_f1'):.4f}",
                    "weighted_f1": f"{bvm.get('weighted_f1'):.4f}",
                    "macro_auc": f"{bvm.get('macro_auc'):.4f}" if bvm.get('macro_auc') is not None else "NA",
                    "loss": f"{bvm.get('loss'):.4f}",
                    "notes": f"Fold {f} of 5-fold CV"
                })
        mean_acc, std_acc = np.mean(f_accs), np.std(f_accs)
        mean_f1, std_f1 = np.mean(f_f1s), np.std(f_f1s)
        tot_time = sum(f_times)
        print(f"  Mean: Acc={mean_acc:.4f} +/- {std_acc:.4f}, Macro-F1={mean_f1:.4f} +/- {std_f1:.4f}, Total Time={tot_time:.1f}s ({tot_time/60:.1f}m)")
        results.append({
            "modality": mod,
            "model": "resnet18",
            "dataset_version": "2.0.0",
            "seed": 42,
            "split_strategy": "stratified_5fold_cv",
            "fold": "MEAN+/-STD",
            "accuracy": f"{mean_acc:.4f}",
            "macro_f1": f"{mean_f1:.4f}",
            "weighted_f1": f"{np.mean([float(r['weighted_f1']) for r in results if r['modality'] == mod]):.4f}",
            "macro_auc": f"{np.mean([float(r['macro_auc']) for r in results if r['modality'] == mod and r['macro_auc'] != 'NA']):.4f}",
            "loss": f"{np.mean([float(r['loss']) for r in results if r['modality'] == mod]):.4f}",
            "notes": f"CV 5-fold mean: Acc={mean_acc:.4f}+/-{std_acc:.4f}, F1={mean_f1:.4f}+/-{std_f1:.4f}"
        })

    # 2. Single-split modalities
    for mod in ["brain_mri", "retina", "skin", "chest_xray", "histopathology"]:
        mod_file = EXP_DIR / mod / "experiment_results.json"
        print(f"\n--- {mod.upper()} (Single Split) ---")
        if mod_file.exists():
            with open(mod_file, "r") as fp:
                data = json.load(fp)
            bvm = data.get("best_val_metrics", {})
            tm = data.get("test_metrics", {})
            time_s = data.get("training_time_seconds", 0)
            best_ep = data.get("best_epoch")
            tot_ep = data.get("total_epochs")
            print(f"  Epochs: {best_ep}/{tot_ep}, Time: {time_s:.1f}s ({time_s/60:.1f}m)")
            print(f"  Val:  Acc={bvm.get('accuracy', 'NA')}, Macro-F1={bvm.get('macro_f1', 'NA')}, Loss={bvm.get('loss', 'NA')}")
            print(f"  Test: Acc={tm.get('accuracy', 'NA')}, Macro-F1={tm.get('macro_f1', 'NA')}, Loss={tm.get('loss', 'NA')}")

            # Val row
            val_acc = f"{bvm.get('accuracy'):.4f}" if isinstance(bvm.get('accuracy'), (int, float)) else "NA"
            val_f1 = f"{bvm.get('macro_f1'):.4f}" if isinstance(bvm.get('macro_f1'), (int, float)) else "NA"
            val_wf1 = f"{bvm.get('weighted_f1'):.4f}" if isinstance(bvm.get('weighted_f1'), (int, float)) else "NA"
            val_auc = f"{bvm.get('macro_auc'):.4f}" if isinstance(bvm.get('macro_auc'), (int, float)) else "NA"
            val_loss = f"{bvm.get('loss'):.4f}" if isinstance(bvm.get('loss'), (int, float)) else "NA"
            results.append({
                "modality": mod,
                "model": "resnet18",
                "dataset_version": "2.0.0",
                "seed": 42,
                "split_strategy": "fixed_validation",
                "fold": "val",
                "accuracy": val_acc,
                "macro_f1": val_f1,
                "weighted_f1": val_wf1,
                "macro_auc": val_auc,
                "loss": val_loss,
                "notes": "Validation split evaluation"
            })

            # Test row
            test_acc = f"{tm.get('accuracy'):.4f}" if isinstance(tm.get('accuracy'), (int, float)) else "NA"
            test_f1 = f"{tm.get('macro_f1'):.4f}" if isinstance(tm.get('macro_f1'), (int, float)) else "NA"
            test_wf1 = f"{tm.get('weighted_f1'):.4f}" if isinstance(tm.get('weighted_f1'), (int, float)) else "NA"
            test_auc = f"{tm.get('macro_auc'):.4f}" if isinstance(tm.get('macro_auc'), (int, float)) else "NA"
            test_loss = f"{tm.get('loss'):.4f}" if isinstance(tm.get('loss'), (int, float)) else "NA"
            results.append({
                "modality": mod,
                "model": "resnet18",
                "dataset_version": "2.0.0",
                "seed": 42,
                "split_strategy": "fixed_test",
                "fold": "test",
                "accuracy": test_acc,
                "macro_f1": test_f1,
                "weighted_f1": test_wf1,
                "macro_auc": test_auc,
                "loss": test_loss,
                "notes": "Held-out test split evaluation (unaltered test partition)"
            })

    # Save cleanly to CSV without any duplicate entries
    clean_df = pd.DataFrame(results)
    clean_df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved {len(clean_df)} clean rows to {CSV_PATH}")

if __name__ == "__main__":
    inspect_all()
